// C quick test for the OpenCL build of libvkfft, fp32 throughout: 1D c2c of
// size 64 forward and inverse, in-place and out-of-place, then a table of r2c
// cases covering both directions in 1D and 2D, even and odd real axes, batches,
// an omitted axis, and a forward-only plan, then a table of fused convolution
// cases, then one save and reload of a compiled plan. Ground truth is a naive
// O(n^2) DFT or a direct summation in double precision computed here.
// Everything the wrapper leaves to the caller (buffer allocation, direction
// encoding, the 1/N factor, synchronization) is done in this file. This doubles
// as a worked example of the wrapper API.

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "vkfft_wrapper.h"

#define FFT_SIZE 64
#define TOLERANCE 1e-5 // on values normalized by FFT_SIZE, so this is a relative bound
#define DIRECTION_FORWARD (-1)
#define DIRECTION_INVERSE 1
#define MAX_R2C_DIM 2
#define MAX_CONV_DIM 2
#define SAVE_LOAD_AXIS 64 // 3D, so the saved blob covers six compiled programs
#define POISON (-987654.0f) // written to an out-of-place destination before every run

static int failures = 0;

static void report(const char* const name, const double err, const double tol) {
    const int ok = (err <= tol) && !isnan(err);
    if (!ok) {
        failures += 1;
    }
    printf("[%s] %-38s max error %.3e (tol %.1e)\n", ok ? "PASS" : "FAIL", name, err, tol);
}

static int fail_cl(const char* const what, const cl_int err) {
    fprintf(stderr, "[FAIL] OpenCL call failed: %s (code %d)\n", what, (int)err);
    return 1;
}

static int fail_vkfft(const char* const what, const int res) {
    fprintf(stderr, "[FAIL] %s: %s (code %d)\n", what, vkfft_error_name(res), res);
    return 1;
}

// Forward DFT with the exp(-2*pi*i*k*n/N) convention, scaled by 1/N so the
// output is around 1.
static void _reference_dft(const float* const in, double* const out, const int n) {
    for (int k = 0; k < n; ++k) {
        double re = 0.0;
        double im = 0.0;
        for (int j = 0; j < n; ++j) {
            const double phase = -2.0 * M_PI * (double)k * (double)j / (double)n;
            const double c = cos(phase);
            const double s = sin(phase);
            re += (double)in[2 * j] * c - (double)in[2 * j + 1] * s;
            im += (double)in[2 * j] * s + (double)in[2 * j + 1] * c;
        }
        out[2 * k] = re / (double)n;
        out[2 * k + 1] = im / (double)n;
    }
}

// Compares a device result against a reference, both taken as complex
// interleaved. The device values are divided by `scale` first.
static double _max_error(const float* const got, const double* const want, const int n, const double scale) {
    double worst = 0.0;
    for (int i = 0; i < 2 * n; ++i) {
        const double diff = fabs((double)got[i] / scale - want[i]);
        if (diff > worst) {
            worst = diff;
        }
    }
    return worst;
}

// One r2c case. Sizes are VkFFT's, so size[0] is the real (contiguous) axis,
// and `batches` stacks on top of every size[] axis.
typedef struct {
    const char* name;
    uint64_t fft_dim;
    uint64_t size[MAX_R2C_DIM];
    uint64_t omit[MAX_R2C_DIM];
    uint64_t batches; // 0 leaves VkFFT's default of 1
    int inplace;
    int forward_only;
} r2c_case;

// Naive DFT along one axis of a dense complex array held as separate real and
// imaginary parts, in place. `n` is the axis length and `stride` the distance
// between neighbours along it, so every base index that is not a line start
// along this axis is skipped.
static void _dft_axis(double* const re, double* const im, const size_t total, const size_t n,
                      const size_t stride) {
    double* const line_re = (double*)malloc(n * sizeof(double));
    double* const line_im = (double*)malloc(n * sizeof(double));
    if (!line_re || !line_im) {
        return;
    }
    for (size_t base = 0; base < total; ++base) {
        if (((base / stride) % n) != 0) {
            continue;
        }
        for (size_t k = 0; k < n; ++k) {
            double sum_re = 0.0;
            double sum_im = 0.0;
            for (size_t j = 0; j < n; ++j) {
                const double phase = -2.0 * M_PI * (double)k * (double)j / (double)n;
                const double c = cos(phase);
                const double s = sin(phase);
                const double xr = re[base + j * stride];
                const double xi = im[base + j * stride];
                sum_re += xr * c - xi * s;
                sum_im += xr * s + xi * c;
            }
            line_re[k] = sum_re;
            line_im[k] = sum_im;
        }
        for (size_t k = 0; k < n; ++k) {
            re[base + k * stride] = line_re[k];
            im[base + k * stride] = line_im[k];
        }
    }
    free(line_re);
    free(line_im);
}

// Runs one r2c case and reports its checks. Allocates its own buffers so each
// case is independent. Returns 1 on a host or OpenCL failure that makes the
// rest of the run meaningless, 0 otherwise (a numerical failure is counted by
// report()).
static int _run_r2c_case(const cl_context context, const cl_command_queue queue,
                         void** const device_handles, const r2c_case* const c) {
    const uint64_t batches = (c->batches == 0) ? 1 : c->batches;
    const size_t real_axis = (size_t)c->size[0];
    const size_t half_axis = real_axis / 2 + 1; // complex values along the real axis

    // Every axis after the first, flattened. Rows are laid out identically on
    // the real and the complex side, so only the length of the first axis
    // differs.
    size_t rows = 1;
    double scale = 1.0; // round-trip factor: the product of the transformed axes
    for (uint64_t a = 0; a < c->fft_dim; ++a) {
        if (a > 0) {
            rows *= (size_t)c->size[a];
        }
        if (c->omit[a] == 0) {
            scale *= (double)c->size[a];
        }
    }
    rows *= (size_t)batches;

    // In place, the real side shares the complex side's row stride, so its rows
    // are padded to 2 * half_axis reals. Out of place the real side is dense.
    const size_t real_row = c->inplace ? (2 * half_axis) : real_axis;
    const size_t real_floats = rows * real_row;
    const size_t cplx_floats = rows * half_axis * 2;
    const size_t dense_floats = rows * real_axis;

    float* const host_real = (float*)malloc(real_floats * sizeof(float));
    float* const host_cplx = (float*)malloc(cplx_floats * sizeof(float));
    float* const host_cplx_after = (float*)malloc(cplx_floats * sizeof(float));
    float* const dense = (float*)malloc(dense_floats * sizeof(float));
    double* const ref_re = (double*)malloc(dense_floats * sizeof(double));
    double* const ref_im = (double*)malloc(dense_floats * sizeof(double));
    if (!host_real || !host_cplx || !host_cplx_after || !dense || !ref_re || !ref_im) {
        fprintf(stderr, "[FAIL] host allocation failed for %s\n", c->name);
        return 1;
    }

    // Deterministic real signal, then the same values in the device's real
    // layout.
    for (size_t i = 0; i < dense_floats; ++i) {
        const double t = (double)i / (double)dense_floats;
        dense[i] = (float)(cos(2.0 * M_PI * 3.0 * t) + 0.5 * sin(2.0 * M_PI * 7.0 * t) - 0.125);
    }
    for (size_t i = 0; i < real_floats; ++i) {
        host_real[i] = POISON;
    }
    for (size_t r = 0; r < rows; ++r) {
        for (size_t i = 0; i < real_axis; ++i) {
            host_real[r * real_row + i] = dense[r * real_axis + i];
        }
    }

    // Reference: a full complex DFT along the transformed axes only, with the
    // omitted axes and the batch axis left as batch axes. The device returns
    // the first half_axis values along the real axis, which is what gets
    // compared.
    for (size_t i = 0; i < dense_floats; ++i) {
        ref_re[i] = (double)dense[i];
        ref_im[i] = 0.0;
    }
    size_t stride = 1;
    for (uint64_t a = 0; a < c->fft_dim; ++a) {
        if (c->omit[a] == 0) {
            _dft_axis(ref_re, ref_im, dense_floats, (size_t)c->size[a], stride);
        }
        stride *= (size_t)c->size[a];
    }

    vkfft_config config = {0};
    config.fft_dim = c->fft_dim;
    for (uint64_t a = 0; a < c->fft_dim; ++a) {
        config.size[a] = c->size[a];
        config.omit[a] = c->omit[a];
    }
    config.number_batches = c->batches;
    config.r2c = 1;
    config.inplace = c->inplace;
    config.make_forward_only = c->forward_only;

    vkfft_app* app = NULL;
    int res = vkfft_create(&config, device_handles, &app);
    if (res != 0) {
        printf("[FAIL] %-38s vkfft_create: %s (code %d)\n", c->name, vkfft_error_name(res), res);
        failures += 1;
        return 0;
    }

    cl_int err = CL_SUCCESS;
    const size_t real_bytes = real_floats * sizeof(float);
    const size_t cplx_bytes = cplx_floats * sizeof(float);
    cl_mem buf_real = clCreateBuffer(context, CL_MEM_READ_WRITE, real_bytes, NULL, &err);
    if (err != CL_SUCCESS) {
        return fail_cl("clCreateBuffer(r2c real)", err);
    }
    // In place, one buffer plays both roles, and it is the complex-sized one.
    cl_mem buf_cplx = buf_real;
    if (!c->inplace) {
        buf_cplx = clCreateBuffer(context, CL_MEM_READ_WRITE, cplx_bytes, NULL, &err);
        if (err != CL_SUCCESS) {
            return fail_cl("clCreateBuffer(r2c complex)", err);
        }
    }

    clEnqueueWriteBuffer(queue, buf_real, CL_TRUE, 0, real_bytes, host_real, 0, NULL, NULL);
    if (!c->inplace) {
        for (size_t i = 0; i < cplx_floats; ++i) {
            host_cplx[i] = POISON;
        }
        clEnqueueWriteBuffer(queue, buf_cplx, CL_TRUE, 0, cplx_bytes, host_cplx, 0, NULL, NULL);
    }

    char label[128];
    res = vkfft_execute(app, buf_real, buf_cplx, DIRECTION_FORWARD, queue);
    if (res != 0) {
        snprintf(label, sizeof(label), "%s forward", c->name);
        failures += fail_vkfft(label, res);
        vkfft_destroy(app);
        return 0;
    }
    clFinish(queue);
    clEnqueueReadBuffer(queue, buf_cplx, CL_TRUE, 0, cplx_bytes, host_cplx, 0, NULL, NULL);

    if (!c->inplace) {
        size_t poisoned = 0;
        for (size_t i = 0; i < cplx_floats; ++i) {
            if (host_cplx[i] == POISON) {
                poisoned += 1;
            }
        }
        snprintf(label, sizeof(label), "%s forward writes every output", c->name);
        report(label, (double)poisoned / (double)cplx_floats, 0.0);
    }

    double worst = 0.0;
    for (size_t r = 0; r < rows; ++r) {
        for (size_t k = 0; k < half_axis; ++k) {
            const size_t dev = 2 * (r * half_axis + k);
            const size_t ref = r * real_axis + k;
            const double dre = fabs((double)host_cplx[dev] / scale - ref_re[ref] / scale);
            const double dim = fabs((double)host_cplx[dev + 1] / scale - ref_im[ref] / scale);
            if (dre > worst) {
                worst = dre;
            }
            if (dim > worst) {
                worst = dim;
            }
        }
    }
    snprintf(label, sizeof(label), "%s forward vs naive DFT", c->name);
    report(label, worst, TOLERANCE);

    if (c->forward_only) {
        // The plan carries no inverse kernels, so VkFFT has to refuse rather
        // than run something. Together with the forward check above this is
        // what says a forward-only plan is unharmed by the inverse-side flag
        // the out-of-place r2c path sets at create time.
        res = vkfft_execute(app, buf_cplx, buf_real, DIRECTION_INVERSE, queue);
        snprintf(label, sizeof(label), "%s inverse refused", c->name);
        report(label, (res != 0) ? 0.0 : 1.0, 0.0);
        printf("       %s inverse returned %s\n", c->name, vkfft_error_name(res));
    }
    else {
        // Invert the spectrum just measured. Out of place the real destination
        // is poisoned first, so an inverse that writes nothing cannot pass.
        if (!c->inplace) {
            for (size_t i = 0; i < real_floats; ++i) {
                host_real[i] = POISON;
            }
            clEnqueueWriteBuffer(queue, buf_real, CL_TRUE, 0, real_bytes, host_real, 0, NULL, NULL);
        }
        res = vkfft_execute(app, buf_cplx, buf_real, DIRECTION_INVERSE, queue);
        if (res != 0) {
            snprintf(label, sizeof(label), "%s inverse", c->name);
            failures += fail_vkfft(label, res);
            vkfft_destroy(app);
            return 0;
        }
        clFinish(queue);
        clEnqueueReadBuffer(queue, buf_real, CL_TRUE, 0, real_bytes, host_real, 0, NULL, NULL);
        clEnqueueReadBuffer(queue, buf_cplx, CL_TRUE, 0, cplx_bytes, host_cplx_after, 0, NULL, NULL);

        if (!c->inplace) {
            size_t poisoned = 0;
            for (size_t r = 0; r < rows; ++r) {
                for (size_t i = 0; i < real_axis; ++i) {
                    if (host_real[r * real_row + i] == POISON) {
                        poisoned += 1;
                    }
                }
            }
            snprintf(label, sizeof(label), "%s inverse writes every output", c->name);
            report(label, (double)poisoned / (double)dense_floats, 0.0);
        }

        worst = 0.0;
        for (size_t r = 0; r < rows; ++r) {
            for (size_t i = 0; i < real_axis; ++i) {
                const double got = (double)host_real[r * real_row + i] / scale;
                const double diff = fabs(got - (double)dense[r * real_axis + i]);
                if (diff > worst) {
                    worst = diff;
                }
            }
        }
        snprintf(label, sizeof(label), "%s round trip", c->name);
        report(label, worst, TOLERANCE);

        // Not a check, a measurement the Julia layer needs: whether the inverse
        // leaves the spectrum it read intact, which decides if a c2r plan can
        // be handed a buffer the caller still wants.
        if (!c->inplace) {
            double spectrum_change = 0.0;
            for (size_t i = 0; i < cplx_floats; ++i) {
                const double d = fabs((double)host_cplx_after[i] - (double)host_cplx[i]) / scale;
                if (d > spectrum_change) {
                    spectrum_change = d;
                }
            }
            printf("       %s spectrum change across the inverse: %.3e\n", c->name, spectrum_change);
        }
    }

    vkfft_destroy(app);
    clReleaseMemObject(buf_real);
    if (!c->inplace) {
        clReleaseMemObject(buf_cplx);
    }
    free(host_real);
    free(host_cplx);
    free(host_cplx_after);
    free(dense);
    free(ref_re);
    free(ref_im);
    return 0;
}

// One fused convolution case, c2c over one or two axes. `features` is VkFFT's
// coordinateFeatures and `kernels` its numberKernels, so the kernel buffer
// holds kernels * features systems and the output holds the same, while the
// input holds `features` of them.
//
// A 1D convolution is written here as fft_dim 2 with size {n, 1} rather than
// fft_dim 1, which is the same transform. See the note in main().
typedef struct {
    const char* name;
    uint64_t fft_dim;
    uint64_t size[MAX_CONV_DIM];
    uint64_t features;
    uint64_t kernels;
    int conjugate;
    int inplace;
} conv_case;

// Circular convolution of `x` with `k` over an n0 by n1 grid, or their circular
// cross-correlation when `conjugate` is set. Everything is interleaved complex
// with the first axis contiguous.
static void _reference_conv(const double* const x, const double* const k, double* const out, const size_t n0,
                            const size_t n1, const int conjugate) {
    for (size_t m1 = 0; m1 < n1; ++m1) {
        for (size_t m0 = 0; m0 < n0; ++m0) {
            double re = 0.0;
            double im = 0.0;
            for (size_t j1 = 0; j1 < n1; ++j1) {
                for (size_t j0 = 0; j0 < n0; ++j0) {
                    // Convolution pairs x[j] with k[m - j], correlation conj(x[j]) with k[j + m].
                    const size_t p0 = conjugate ? ((j0 + m0) % n0) : ((m0 + n0 - j0) % n0);
                    const size_t p1 = conjugate ? ((j1 + m1) % n1) : ((m1 + n1 - j1) % n1);
                    const size_t j = j1 * n0 + j0;
                    const size_t p = p1 * n0 + p0;
                    const double xr = x[2 * j];
                    const double xi = conjugate ? -x[2 * j + 1] : x[2 * j + 1];
                    re += xr * k[2 * p] - xi * k[2 * p + 1];
                    im += xr * k[2 * p + 1] + xi * k[2 * p];
                }
            }
            out[2 * (m1 * n0 + m0)] = re;
            out[2 * (m1 * n0 + m0) + 1] = im;
        }
    }
}

// Runs one convolution case: builds the kernel plan, transforms the kernel on
// the device, builds the convolution plan pointing at it, runs it, and compares
// against the host reference. Returns 1 only on a failure that makes the rest
// of the run meaningless.
static int _run_conv_case(const cl_context context, const cl_command_queue queue, void** const device_handles,
                          const conv_case* const c) {
    const size_t n0 = (size_t)c->size[0];
    const size_t n1 = (c->fft_dim > 1) ? (size_t)c->size[1] : 1;
    const size_t n = n0 * n1;
    const size_t systems_in = (size_t)c->features;
    const size_t systems_out = (size_t)(c->kernels * c->features);
    const size_t floats_in = 2 * n * systems_in;
    const size_t floats_out = 2 * n * systems_out;

    float* const host_in = (float*)malloc(floats_in * sizeof(float));
    float* const host_kernel = (float*)malloc(floats_out * sizeof(float));
    float* const host_out = (float*)malloc(floats_out * sizeof(float));
    double* const ref_in = (double*)malloc(floats_in * sizeof(double));
    double* const ref_kernel = (double*)malloc(floats_out * sizeof(double));
    double* const ref_out = (double*)malloc(2 * n * sizeof(double));
    if (!host_in || !host_kernel || !host_out || !ref_in || !ref_kernel || !ref_out) {
        fprintf(stderr, "[FAIL] host allocation failed for %s\n", c->name);
        return 1;
    }

    // Deterministic input, one distinct signal per feature component.
    for (size_t s = 0; s < systems_in; ++s) {
        for (size_t i = 0; i < n; ++i) {
            const double t = (double)i / (double)n;
            const double f = 3.0 + (double)s;
            ref_in[2 * (s * n + i)] = cos(2.0 * M_PI * f * t) + 0.25;
            ref_in[2 * (s * n + i) + 1] = 0.5 * sin(2.0 * M_PI * (f + 4.0) * t);
        }
    }
    // A short, decaying kernel, again distinct per system, so the convolution
    // mixes several taps rather than acting as a scaled identity. The taps sit
    // in the first few positions of the first axis of each system.
    for (size_t s = 0; s < systems_out; ++s) {
        for (size_t i = 0; i < n; ++i) {
            const size_t along0 = i % n0;
            const size_t along1 = i / n0;
            const double amp = ((along0 < 5) && (along1 < 2)) ? (1.0 / (double)(along0 + 2 * along1 + 1)) : 0.0;
            ref_kernel[2 * (s * n + i)] = amp * (1.0 + 0.25 * (double)s);
            ref_kernel[2 * (s * n + i) + 1] = -0.5 * amp;
        }
    }
    for (size_t i = 0; i < floats_in; ++i) {
        host_in[i] = (float)ref_in[i];
    }
    for (size_t i = 0; i < floats_out; ++i) {
        host_kernel[i] = (float)ref_kernel[i];
    }

    cl_int err = CL_SUCCESS;
    const size_t bytes_in = floats_in * sizeof(float);
    const size_t bytes_out = floats_out * sizeof(float);
    cl_mem buf_kernel = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes_out, NULL, &err);
    if (err != CL_SUCCESS) {
        return fail_cl("clCreateBuffer(conv kernel)", err);
    }
    // In place, one buffer is both input and output, so it has to be the
    // output-sized one. That only fits when there is a single kernel.
    cl_mem buf_out = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes_out, NULL, &err);
    if (err != CL_SUCCESS) {
        return fail_cl("clCreateBuffer(conv out)", err);
    }
    cl_mem buf_in = buf_out;
    if (!c->inplace) {
        buf_in = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes_in, NULL, &err);
        if (err != CL_SUCCESS) {
            return fail_cl("clCreateBuffer(conv in)", err);
        }
    }

    char label[160];

    // 1. The kernel plan. Its batch axis is what stacks the kernels, so it runs
    // with number_batches = numberKernels and the convolution plan reads the
    // result back with number_kernels set instead.
    vkfft_config kernel_config = {0};
    kernel_config.fft_dim = c->fft_dim;
    for (uint64_t a = 0; a < c->fft_dim; ++a) {
        kernel_config.size[a] = c->size[a];
    }
    kernel_config.inplace = 1;
    kernel_config.kernel_convolution = 1;
    kernel_config.coordinate_features = c->features;
    kernel_config.number_batches = c->kernels;
    kernel_config.make_forward_only = 1;

    vkfft_app* kernel_app = NULL;
    int res = vkfft_create(&kernel_config, device_handles, &kernel_app);
    if (res != 0) {
        printf("[FAIL] %-42s vkfft_create (kernel): %s (code %d)\n", c->name, vkfft_error_name(res), res);
        failures += 1;
        return 0;
    }

    clEnqueueWriteBuffer(queue, buf_kernel, CL_TRUE, 0, bytes_out, host_kernel, 0, NULL, NULL);
    res = vkfft_execute(kernel_app, buf_kernel, buf_kernel, DIRECTION_FORWARD, queue);
    if (res != 0) {
        snprintf(label, sizeof(label), "%s kernel transform", c->name);
        failures += fail_vkfft(label, res);
        vkfft_destroy(kernel_app);
        return 0;
    }
    clFinish(queue);
    vkfft_destroy(kernel_app); // the plan is done, the transformed kernel stays in the buffer

    // 2. The convolution plan, same layout, pointed at that buffer.
    vkfft_config conv_config = {0};
    conv_config.fft_dim = c->fft_dim;
    for (uint64_t a = 0; a < c->fft_dim; ++a) {
        conv_config.size[a] = c->size[a];
    }
    conv_config.inplace = c->inplace;
    conv_config.perform_convolution = 1;
    conv_config.coordinate_features = c->features;
    conv_config.number_kernels = c->kernels;
    conv_config.conjugate_convolution = c->conjugate;
    // The convolution step normalizes its own axis unconditionally, but only
    // that one, so a multi-axis pipeline comes back scaled by the product of
    // the other axes. normalize covers every inverse axis instead, and the
    // convolution axis is not scaled twice because VkFFT applies it once
    // either way, so this is what makes the output the plain convolution.
    conv_config.normalize = 1;
    // No make_forward_only here. A convolution plan runs forward, multiplies,
    // then inverse, so it needs both halves even though it is only ever
    // executed in the forward direction.

    vkfft_app* conv_app = NULL;
    res = vkfft_create(&conv_config, device_handles, &conv_app);
    if (res != 0) {
        printf("[FAIL] %-42s vkfft_create (conv): %s (code %d)\n", c->name, vkfft_error_name(res), res);
        failures += 1;
        return 0;
    }

    // Executing before the kernel slot is filled has to be refused rather than
    // reach the driver as a null buffer argument.
    res = vkfft_execute(conv_app, buf_in, buf_out, DIRECTION_FORWARD, queue);
    snprintf(label, sizeof(label), "%s refused without a kernel", c->name);
    report(label, (res == 2012) ? 0.0 : 1.0, 0.0);

    res = vkfft_set_kernel(conv_app, buf_kernel);
    if (res != 0) {
        snprintf(label, sizeof(label), "%s vkfft_set_kernel", c->name);
        failures += fail_vkfft(label, res);
        vkfft_destroy(conv_app);
        return 0;
    }

    // Poison the destination so a plan that writes nothing cannot pass, then
    // put the input in place. In place the two are the same buffer, so the
    // input has to go in second.
    for (size_t i = 0; i < floats_out; ++i) {
        host_out[i] = POISON;
    }
    clEnqueueWriteBuffer(queue, buf_out, CL_TRUE, 0, bytes_out, host_out, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, buf_in, CL_TRUE, 0, bytes_in, host_in, 0, NULL, NULL);

    res = vkfft_execute(conv_app, buf_in, buf_out, DIRECTION_FORWARD, queue);
    if (res != 0) {
        snprintf(label, sizeof(label), "%s convolve", c->name);
        failures += fail_vkfft(label, res);
        vkfft_destroy(conv_app);
        return 0;
    }
    clFinish(queue);
    clEnqueueReadBuffer(queue, buf_out, CL_TRUE, 0, bytes_out, host_out, 0, NULL, NULL);

    size_t poisoned = 0;
    for (size_t i = 0; i < floats_out; ++i) {
        if (host_out[i] == POISON) {
            poisoned += 1;
        }
    }
    snprintf(label, sizeof(label), "%s writes every output", c->name);
    report(label, (double)poisoned / (double)floats_out, 0.0);

    // The output stacks kernels outside features, matching how the kernel plan
    // stacked its batches outside them.
    double worst = 0.0;
    for (uint64_t kk = 0; kk < c->kernels; ++kk) {
        for (uint64_t v = 0; v < c->features; ++v) {
            const size_t sys = (size_t)(kk * c->features + v);
            _reference_conv(&ref_in[2 * n * (size_t)v], &ref_kernel[2 * n * sys], ref_out, n0, n1, c->conjugate);
            for (size_t i = 0; i < 2 * n; ++i) {
                const double diff = fabs((double)host_out[2 * n * sys + i] - ref_out[i]) / (double)n;
                if (diff > worst) {
                    worst = diff;
                }
            }
        }
    }
    snprintf(label, sizeof(label), "%s vs direct summation", c->name);
    report(label, worst, TOLERANCE);

    vkfft_destroy(conv_app);
    clReleaseMemObject(buf_kernel);
    clReleaseMemObject(buf_out);
    if (!c->inplace) {
        clReleaseMemObject(buf_in);
    }
    free(host_in);
    free(host_kernel);
    free(host_out);
    free(ref_in);
    free(ref_kernel);
    free(ref_out);
    return 0;
}

// Seconds on a monotonic-enough clock. Only the difference of two of these is
// used, and only to show that reloading a plan skips the compile.
static double _now(void) {
    struct timespec ts;
    if (timespec_get(&ts, TIME_UTC) != TIME_UTC) {
        return 0.0;
    }
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

// Creates a plan that keeps its binaries, saves them, throws the plan away,
// rebuilds it from the saved bytes and checks that the rebuilt plan produces
// the identical result. Returns 1 only on a failure that makes the rest of the
// run meaningless.
static int _run_save_load(const cl_context context, const cl_command_queue queue, void** const device_handles) {
    // A 3D plan with both directions built, so the saved blob covers six
    // compiled programs rather than one. A single small plan compiles fast
    // enough that the reload saves no measurable time, which tells nobody
    // anything.
    const size_t axis = SAVE_LOAD_AXIS;
    const size_t n = axis * axis * axis;
    const size_t bytes = 2 * n * sizeof(float);
    float* const host_in = (float*)malloc(bytes);
    float* const fresh = (float*)malloc(bytes);
    float* const loaded = (float*)malloc(bytes);
    if (!host_in || !fresh || !loaded) {
        fprintf(stderr, "[FAIL] host allocation failed for save/load\n");
        return 1;
    }
    for (size_t i = 0; i < n; ++i) {
        const double t = (double)i / (double)n;
        host_in[2 * i] = (float)(cos(2.0 * M_PI * 3.0 * t) + 0.25);
        host_in[2 * i + 1] = (float)(0.5 * sin(2.0 * M_PI * 11.0 * t));
    }

    cl_int err = CL_SUCCESS;
    cl_mem buf_in = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, NULL, &err);
    if (err != CL_SUCCESS) {
        return fail_cl("clCreateBuffer(save in)", err);
    }
    cl_mem buf_out = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, NULL, &err);
    if (err != CL_SUCCESS) {
        return fail_cl("clCreateBuffer(save out)", err);
    }
    clEnqueueWriteBuffer(queue, buf_in, CL_TRUE, 0, bytes, host_in, 0, NULL, NULL);

    vkfft_config config = {0};
    config.fft_dim = 3;
    config.size[0] = axis;
    config.size[1] = axis;
    config.size[2] = axis;
    config.save_to_string = 1;

    const double t0 = _now();
    vkfft_app* app = NULL;
    int res = vkfft_create(&config, device_handles, &app);
    const double compile_seconds = _now() - t0;
    if (res != 0) {
        return fail_vkfft("vkfft_create (save_to_string)", res);
    }

    res = vkfft_execute(app, buf_in, buf_out, DIRECTION_FORWARD, queue);
    if (res != 0) {
        return fail_vkfft("vkfft_execute (compiled plan)", res);
    }
    clFinish(queue);
    clEnqueueReadBuffer(queue, buf_out, CL_TRUE, 0, bytes, fresh, 0, NULL, NULL);

    const char* saved = NULL;
    uint64_t saved_size = 0;
    res = vkfft_save(app, &saved, &saved_size);
    if (res != 0) {
        return fail_vkfft("vkfft_save", res);
    }
    // The bytes belong to the app and die with it, so they are copied out
    // before the app goes away. This is exactly what a caller that wants a
    // cache on disk has to do.
    char* const blob = (char*)malloc((size_t)saved_size);
    if (!blob) {
        fprintf(stderr, "[FAIL] host allocation failed for the saved blob\n");
        return 1;
    }
    memcpy(blob, saved, (size_t)saved_size);
    vkfft_destroy(app);

    printf("       saved %llu bytes of compiled binaries\n", (unsigned long long)saved_size);
    report("saved blob is non-empty", (saved_size > 0) ? 0.0 : 1.0, 0.0);

    // A blob shorter than its own header claims has to be refused rather than
    // read past the end.
    vkfft_app* truncated_app = NULL;
    res = vkfft_create_loaded(&config, device_handles, blob, saved_size / 2, &truncated_app);
    report("truncated blob refused", (res == -1) ? 0.0 : 1.0, 0.0);

    // The reloaded plan must not ask to save as well, which VkFFT refuses.
    vkfft_config load_config = config;
    load_config.save_to_string = 0;

    for (size_t i = 0; i < 2 * n; ++i) {
        loaded[i] = POISON;
    }
    clEnqueueWriteBuffer(queue, buf_out, CL_TRUE, 0, bytes, loaded, 0, NULL, NULL);

    const double t1 = _now();
    vkfft_app* reloaded = NULL;
    res = vkfft_create_loaded(&load_config, device_handles, blob, saved_size, &reloaded);
    const double load_seconds = _now() - t1;
    if (res != 0) {
        return fail_vkfft("vkfft_create_loaded", res);
    }
    free(blob); // VkFFT reads it during the create call only

    res = vkfft_execute(reloaded, buf_in, buf_out, DIRECTION_FORWARD, queue);
    if (res != 0) {
        return fail_vkfft("vkfft_execute (reloaded plan)", res);
    }
    clFinish(queue);
    clEnqueueReadBuffer(queue, buf_out, CL_TRUE, 0, bytes, loaded, 0, NULL, NULL);

    double worst = 0.0;
    size_t poisoned = 0;
    for (size_t i = 0; i < 2 * n; ++i) {
        if (loaded[i] == POISON) {
            poisoned += 1;
        }
        const double diff = fabs((double)loaded[i] - (double)fresh[i]) / (double)n;
        if (diff > worst) {
            worst = diff;
        }
    }
    report("reloaded plan writes every output", (double)poisoned / (double)(2 * n), 0.0);
    // Same binaries on the same device, so this is an equality check, not an
    // approximation.
    report("reloaded plan matches the compiled one", worst, 0.0);
    // The absolute times depend on whether the driver already has these
    // programs in its own on-disk cache, so the ratio is large on a cold cache
    // and near 1 on a warm one. Only the cold number says what loading saves.
    printf("       vkfft_create %.4f s, vkfft_create_loaded %.4f s (%.1fx, driver cache state affects this)\n",
           compile_seconds, load_seconds, (load_seconds > 0.0) ? (compile_seconds / load_seconds) : 0.0);

    // A plan that was not asked to save has nothing to hand back.
    const char* nothing = NULL;
    uint64_t nothing_size = 0;
    res = vkfft_save(reloaded, &nothing, &nothing_size);
    report("vkfft_save on a plan that did not save", (res == 2013) ? 0.0 : 1.0, 0.0);
    printf("       vkfft_save on a non-saving plan returned %s\n", vkfft_error_name(res));

    vkfft_destroy(reloaded);
    clReleaseMemObject(buf_in);
    clReleaseMemObject(buf_out);
    free(host_in);
    free(fresh);
    free(loaded);
    return 0;
}

int main(void) {
    printf("libvkfft quick test: backend %llu, max dims %llu\n",
           (unsigned long long)vkfft_backend(), (unsigned long long)vkfft_max_dims());
    if (vkfft_backend() != 3) {
        fprintf(stderr, "[FAIL] this test needs the OpenCL build (VKFFT_BACKEND=3)\n");
        return 1;
    }
    report("vkfft_config_size matches this mirror", (vkfft_config_size() == sizeof(vkfft_config)) ? 0.0 : 1.0, 0.0);

    cl_platform_id platform = NULL;
    cl_uint num_platforms = 0;
    cl_int err = clGetPlatformIDs(1, &platform, &num_platforms);
    if ((err != CL_SUCCESS) || (num_platforms == 0)) {
        return fail_cl("clGetPlatformIDs", err);
    }

    cl_device_id device = NULL;
    cl_uint num_devices = 0;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, &num_devices);
    if ((err != CL_SUCCESS) || (num_devices == 0)) {
        return fail_cl("clGetDeviceIDs", err);
    }

    char device_name[256] = {0};
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
    printf("device: %s\n", device_name);

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        return fail_cl("clCreateContext", err);
    }
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    if (err != CL_SUCCESS) {
        return fail_cl("clCreateCommandQueue", err);
    }

    const size_t bytes = 2 * FFT_SIZE * sizeof(float);
    float* const host_in = (float*)malloc(bytes);
    float* const host_out = (float*)malloc(bytes);
    double* const reference = (double*)malloc(2 * FFT_SIZE * sizeof(double));
    double* const delta_reference = (double*)malloc(2 * FFT_SIZE * sizeof(double));
    double* const input_reference = (double*)malloc(2 * FFT_SIZE * sizeof(double));
    if (!host_in || !host_out || !reference || !delta_reference || !input_reference) {
        fprintf(stderr, "[FAIL] host allocation failed\n");
        return 1;
    }

    // A mixed signal, deterministic so the printed errors are reproducible.
    for (int i = 0; i < FFT_SIZE; ++i) {
        const double t = (double)i / (double)FFT_SIZE;
        host_in[2 * i] = (float)(cos(2.0 * M_PI * 3.0 * t) + 0.25 * sin(2.0 * M_PI * 11.0 * t));
        host_in[2 * i + 1] = (float)(0.5 * cos(2.0 * M_PI * 7.0 * t) - 0.125);
        input_reference[2 * i] = (double)host_in[2 * i];
        input_reference[2 * i + 1] = (double)host_in[2 * i + 1];
    }
    _reference_dft(host_in, reference, FFT_SIZE);

    cl_mem buf_in = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, NULL, &err);
    if (err != CL_SUCCESS) {
        return fail_cl("clCreateBuffer(in)", err);
    }
    cl_mem buf_out = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, NULL, &err);
    if (err != CL_SUCCESS) {
        return fail_cl("clCreateBuffer(out)", err);
    }

    void* device_handles[4];
    device_handles[0] = &platform;
    device_handles[1] = &device;
    device_handles[2] = &context;
    device_handles[3] = &queue;

    // Out-of-place plan, both directions.
    vkfft_config config = {0};
    config.fft_dim = 1;
    config.size[0] = FFT_SIZE;

    vkfft_app* app = NULL;
    int res = vkfft_create(&config, device_handles, &app);
    if (res != 0) {
        return fail_vkfft("vkfft_create (out-of-place)", res);
    }

    err = clEnqueueWriteBuffer(queue, buf_in, CL_TRUE, 0, bytes, host_in, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        return fail_cl("clEnqueueWriteBuffer", err);
    }

    res = vkfft_execute(app, buf_in, buf_out, DIRECTION_FORWARD, queue);
    if (res != 0) {
        return fail_vkfft("vkfft_execute (forward)", res);
    }
    err = clFinish(queue);
    if (err != CL_SUCCESS) {
        return fail_cl("clFinish (forward)", err);
    }
    err = clEnqueueReadBuffer(queue, buf_out, CL_TRUE, 0, bytes, host_out, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        return fail_cl("clEnqueueReadBuffer (forward)", err);
    }
    report("forward vs naive DFT", _max_error(host_out, reference, FFT_SIZE, FFT_SIZE), TOLERANCE);

    // Inverse of the forward result: VkFFT's unnormalized inverse returns N*x,
    // so the 1/N is applied here. (config.normalize would do it in-kernel,
    // which is the caller's call.)
    res = vkfft_execute(app, buf_out, buf_in, DIRECTION_INVERSE, queue);
    if (res != 0) {
        return fail_vkfft("vkfft_execute (inverse)", res);
    }
    err = clFinish(queue);
    if (err != CL_SUCCESS) {
        return fail_cl("clFinish (inverse)", err);
    }
    err = clEnqueueReadBuffer(queue, buf_in, CL_TRUE, 0, bytes, host_out, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        return fail_cl("clEnqueueReadBuffer (inverse)", err);
    }
    report("out-of-place round trip", _max_error(host_out, input_reference, FFT_SIZE, FFT_SIZE), TOLERANCE);

    // Single-frequency input: the forward transform is analytically N*delta[k -
    // k0], which also pins down the sign convention of DIRECTION_FORWARD.
    const int k0 = 5;
    for (int i = 0; i < FFT_SIZE; ++i) {
        const double phase = 2.0 * M_PI * (double)k0 * (double)i / (double)FFT_SIZE;
        host_in[2 * i] = (float)cos(phase);
        host_in[2 * i + 1] = (float)sin(phase);
        delta_reference[2 * i] = (i == k0) ? 1.0 : 0.0;
        delta_reference[2 * i + 1] = 0.0;
    }
    err = clEnqueueWriteBuffer(queue, buf_in, CL_TRUE, 0, bytes, host_in, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        return fail_cl("clEnqueueWriteBuffer (delta)", err);
    }
    res = vkfft_execute(app, buf_in, buf_out, DIRECTION_FORWARD, queue);
    if (res != 0) {
        return fail_vkfft("vkfft_execute (delta forward)", res);
    }
    err = clFinish(queue);
    if (err != CL_SUCCESS) {
        return fail_cl("clFinish (delta)", err);
    }
    err = clEnqueueReadBuffer(queue, buf_out, CL_TRUE, 0, bytes, host_out, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        return fail_cl("clEnqueueReadBuffer (delta)", err);
    }
    report("forward of exp(+2*pi*i*5*n/N) is delta[5]",
           _max_error(host_out, delta_reference, FFT_SIZE, FFT_SIZE), TOLERANCE);

    vkfft_destroy(app);

    // In-place plan on a single buffer.
    config.inplace = 1;
    app = NULL;
    res = vkfft_create(&config, device_handles, &app);
    if (res != 0) {
        return fail_vkfft("vkfft_create (in-place)", res);
    }

    for (int i = 0; i < FFT_SIZE; ++i) {
        host_in[2 * i] = (float)input_reference[2 * i];
        host_in[2 * i + 1] = (float)input_reference[2 * i + 1];
    }
    err = clEnqueueWriteBuffer(queue, buf_in, CL_TRUE, 0, bytes, host_in, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        return fail_cl("clEnqueueWriteBuffer (in-place)", err);
    }
    res = vkfft_execute(app, buf_in, buf_in, DIRECTION_FORWARD, queue);
    if (res != 0) {
        return fail_vkfft("vkfft_execute (in-place forward)", res);
    }
    err = clFinish(queue);
    if (err != CL_SUCCESS) {
        return fail_cl("clFinish (in-place forward)", err);
    }
    err = clEnqueueReadBuffer(queue, buf_in, CL_TRUE, 0, bytes, host_out, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        return fail_cl("clEnqueueReadBuffer (in-place forward)", err);
    }
    report("in-place forward vs naive DFT", _max_error(host_out, reference, FFT_SIZE, FFT_SIZE), TOLERANCE);

    res = vkfft_execute(app, buf_in, buf_in, DIRECTION_INVERSE, queue);
    if (res != 0) {
        return fail_vkfft("vkfft_execute (in-place inverse)", res);
    }
    err = clFinish(queue);
    if (err != CL_SUCCESS) {
        return fail_cl("clFinish (in-place inverse)", err);
    }
    err = clEnqueueReadBuffer(queue, buf_in, CL_TRUE, 0, bytes, host_out, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        return fail_cl("clEnqueueReadBuffer (in-place inverse)", err);
    }
    report("in-place round trip", _max_error(host_out, input_reference, FFT_SIZE, FFT_SIZE), TOLERANCE);

    vkfft_destroy(app);

    // r2c, the layout the rfft family needs. Out of place the real side and the
    // complex side have different lengths, and which handle holds which follows
    // the direction, so both directions are checked for every case.
    static const r2c_case r2c_cases[] = {
        {"r2c 1d 16", 1, {16, 0}, {0, 0}, 0, 0, 0},
        {"r2c 1d 13 (odd)", 1, {13, 0}, {0, 0}, 0, 0, 0},
        {"r2c 1d 16 batch 3", 1, {16, 0}, {0, 0}, 3, 0, 0},
        {"r2c 1d 13 batch 3 (odd)", 1, {13, 0}, {0, 0}, 3, 0, 0},
        {"r2c 2d 16x13", 2, {16, 13}, {0, 0}, 0, 0, 0},
        {"r2c 2d 13x8 (odd)", 2, {13, 8}, {0, 0}, 0, 0, 0},
        {"r2c 2d 16x13 batch 2", 2, {16, 13}, {0, 0}, 2, 0, 0},
        {"r2c 2d 16x13 omit2", 2, {16, 13}, {0, 1}, 0, 0, 0},
        {"r2c 1d 16 in-place", 1, {16, 0}, {0, 0}, 0, 1, 0},
        {"r2c 2d 16x13 in-place", 2, {16, 13}, {0, 0}, 0, 1, 0},
        {"r2c 1d 16 forward-only", 1, {16, 0}, {0, 0}, 0, 0, 1},
        {"r2c 2d 16x13 forward-only", 2, {16, 13}, {0, 0}, 0, 0, 1},
    };
    for (size_t i = 0; i < sizeof(r2c_cases) / sizeof(r2c_cases[0]); ++i) {
        if (_run_r2c_case(context, queue, device_handles, &r2c_cases[i]) != 0) {
            return 1;
        }
    }

    // A convolution plan with fft_dim 1 over a size VkFFT transforms in one
    // upload does not compile: its generated code opens a bounds check per
    // register without ever closing it, and the driver rejects the source.
    // Writing the same transform as fft_dim 2 with a trailing axis of length 1
    // moves the convolution onto the strided code path, which is correct, so
    // that is the shape the 1D cases below use. Everything from roughly 8192
    // upward compiles as fft_dim 1 because it needs several uploads and takes
    // the same strided path. This probe records what the vendored VkFFT
    // actually does rather than asserting it, so a fixed upstream shows up as
    // a changed line and not as a failure.
    vkfft_config probe_config = {0};
    probe_config.fft_dim = 1;
    probe_config.size[0] = 64;
    probe_config.perform_convolution = 1;
    vkfft_app* probe_app = NULL;
    res = vkfft_create(&probe_config, device_handles, &probe_app);
    printf("note: fft_dim 1 single-upload convolution returns %s\n", vkfft_error_name(res));
    vkfft_destroy(probe_app);

    // Fused convolution. Each case builds its own kernel plan, transforms the
    // kernel on the device, and runs the convolution plan against it.
    static const conv_case conv_cases[] = {
        {"conv 1d 64", 2, {64, 1}, 1, 1, 0, 0},
        {"conv 1d 64 in-place", 2, {64, 1}, 1, 1, 0, 1},
        {"cross-correlation 1d 64", 2, {64, 1}, 1, 1, 1, 0},
        {"cross-correlation 1d 64 in-place", 2, {64, 1}, 1, 1, 1, 1},
        {"conv 1d 64 features 2", 2, {64, 1}, 2, 1, 0, 0},
        {"conv 2d 16x16", 2, {16, 16}, 1, 1, 0, 0},
        {"cross-correlation 2d 16x16", 2, {16, 16}, 1, 1, 1, 0},
        {"conv 2d 16x8 features 2", 2, {16, 8}, 2, 1, 0, 0},
        // number_kernels needs a genuinely multi-axis shape. Combined with the
        // trailing length-1 axis the 1D cases above use, VkFFT writes zeros
        // instead of a convolution, so these cases carry a real second axis.
        {"conv 2d 16x16 kernels 2", 2, {16, 16}, 1, 2, 0, 0},
        {"conv 2d 16x16 features 2 kernels 2", 2, {16, 16}, 2, 2, 0, 0},
        {"cross-correlation 2d 16x16 kernels 2", 2, {16, 16}, 1, 2, 1, 0},
        {"conv 2d 16x16 kernels 2 in-place", 2, {16, 16}, 1, 2, 0, 1},
    };
    for (size_t i = 0; i < sizeof(conv_cases) / sizeof(conv_cases[0]); ++i) {
        if (_run_conv_case(context, queue, device_handles, &conv_cases[i]) != 0) {
            return 1;
        }
    }

    if (_run_save_load(context, queue, device_handles) != 0) {
        return 1;
    }

    // The error wrapper must come from VkFFT's own table, not from anything
    // written here.
    printf("vkfft_error_name(0) = %s\n", vkfft_error_name(0));
    printf("vkfft_error_name(2004) = %s\n", vkfft_error_name(2004));

    clReleaseMemObject(buf_in);
    clReleaseMemObject(buf_out);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(host_in);
    free(host_out);
    free(reference);
    free(delta_reference);
    free(input_reference);

    printf("%s (%d failing checks)\n", (failures == 0) ? "ALL CHECKS PASSED" : "FAILURES", failures);
    return (failures == 0) ? 0 : 1;
}
