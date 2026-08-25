// C quick test for the OpenCL build of libvkfft, fp32 throughout: 1D c2c of
// size 64 forward and inverse, in-place and out-of-place, then a table of r2c
// cases covering both directions in 1D and 2D, even and odd real axes, batches,
// an omitted axis, and a forward-only plan. Ground truth is a naive O(n^2) DFT
// in double precision computed here. Everything the wrapper leaves to the
// caller (buffer allocation, direction encoding, the 1/N factor,
// synchronization) is done in this file. This doubles as a worked example of
// the wrapper API.

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

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

int main(void) {
    printf("libvkfft quick test: backend %llu, max dims %llu\n",
           (unsigned long long)vkfft_backend(), (unsigned long long)vkfft_max_dims());
    if (vkfft_backend() != 3) {
        fprintf(stderr, "[FAIL] this test needs the OpenCL build (VKFFT_BACKEND=3)\n");
        return 1;
    }

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
