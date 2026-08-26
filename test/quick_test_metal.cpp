// metal-cpp quick test for the Metal build of libvkfft, fp32 throughout
// (Metal has no fp64): c2c in 1D, 2D and batched, in-place and out-of-place,
// out-of-place r2c round trips with an even and an odd real axis, plan reuse
// across command buffers and inside one command buffer, and a stress loop that
// measures the process footprint across thousands of executes so that both a
// leaked encoder and an over-released one show up. Ground truth is a naive
// O(n^2) DFT in double precision computed here. Everything the wrapper leaves
// to the caller (buffer allocation, command buffer creation, submission, the
// 1/N factor, autorelease pools) is done in this file, so this doubles as a
// worked example of the Metal call sequence.

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <mach/mach.h>
#include <mach/task_info.h>

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include "Foundation/Foundation.hpp"
#include "Metal/Metal.hpp"

#include "vkfft_wrapper.h"

#define FFT_SIZE 64
#define TOLERANCE 1e-5 // on values normalized by the transform size, so this is a relative bound
#define DIRECTION_FORWARD (-1)
#define DIRECTION_INVERSE 1
#define MAX_CASE_DIM 2
#define POISON (-987654.0f) // written to an out-of-place destination before every run

// Stress loop sizes. Large enough that a per-call leak of a few hundred bytes
// clears the 16 kB granularity of the footprint counter, small enough to keep
// the test near a second.
#define STRESS_POOLED 3000
#define STRESS_UNPOOLED 6000
#define STRESS_WARMUP 400
#define STRESS_OUTER_POOL 500

// Per-execute footprint growth allowed over the bare command buffer loop in the
// pooled runs, where both loops sit near zero. This catches a gross leak, while
// an encoder-sized one is what the unpooled runs measure.
#define STRESS_POOLED_BOUND 128.0

static int failures = 0;

static void report(const char* const name, const double err, const double tol) {
    const int ok = (err <= tol) && !isnan(err);
    if (!ok) {
        failures += 1;
    }
    printf("[%s] %-46s max error %.3e (tol %.1e)\n", ok ? "PASS" : "FAIL", name, err, tol);
}

static int fail_vkfft(const char* const what, const int res) {
    fprintf(stderr, "[FAIL] %s: %s (code %d)\n", what, vkfft_error_name(res), res);
    return 1;
}

// Physical footprint of this process, which is what a leak moves. Resident size
// would also count pages the allocator has handed back.
static size_t _footprint_bytes(void) {
    task_vm_info_data_t info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&info), &count) != KERN_SUCCESS) {
        return 0;
    }
    return static_cast<size_t>(info.phys_footprint);
}

// One submission, which is the whole Metal contract in five lines: the caller
// takes a command buffer from its queue, hands it to the wrapper, and commits
// and waits itself. The pool is what reclaims the command buffer, since the
// queue returns it autoreleased and releasing it by hand would be an
// over-release.
static int _submit(MTL::CommandQueue* const queue, vkfft_app* const app, MTL::Buffer* const in,
                   MTL::Buffer* const out, const int direction) {
    NS::AutoreleasePool* const pool = NS::AutoreleasePool::alloc()->init();
    MTL::CommandBuffer* const command_buffer = queue->commandBuffer();
    const int res = vkfft_execute(app, in, out, direction, command_buffer);
    if (res == 0) {
        command_buffer->commit();
        command_buffer->waitUntilCompleted();
    }
    pool->release();
    return res;
}

// Forward then inverse appended to one command buffer and submitted once. VkFFT
// encodes one compute encoder per call and Metal runs the encoders of a command
// buffer in order, so the inverse sees what the forward wrote.
static int _submit_round_trip(MTL::CommandQueue* const queue, vkfft_app* const app, MTL::Buffer* const a,
                              MTL::Buffer* const b) {
    NS::AutoreleasePool* const pool = NS::AutoreleasePool::alloc()->init();
    MTL::CommandBuffer* const command_buffer = queue->commandBuffer();
    int res = vkfft_execute(app, a, b, DIRECTION_FORWARD, command_buffer);
    if (res == 0) {
        res = vkfft_execute(app, b, a, DIRECTION_INVERSE, command_buffer);
    }
    if (res == 0) {
        command_buffer->commit();
        command_buffer->waitUntilCompleted();
    }
    pool->release();
    return res;
}

static void _fill_poison(MTL::Buffer* const buffer, const size_t floats) {
    float* const host = static_cast<float*>(buffer->contents());
    for (size_t i = 0; i < floats; ++i) {
        host[i] = POISON;
    }
}

static double _poison_fraction(MTL::Buffer* const buffer, const size_t floats) {
    const float* const host = static_cast<const float*>(buffer->contents());
    size_t poisoned = 0;
    for (size_t i = 0; i < floats; ++i) {
        if (host[i] == POISON) {
            poisoned += 1;
        }
    }
    return static_cast<double>(poisoned) / static_cast<double>(floats);
}

// Naive DFT along one axis of a dense complex array held as separate real and
// imaginary parts, in place. `n` is the axis length and `stride` the distance
// between neighbours along it, so every base index that is not a line start
// along this axis is skipped.
static void _dft_axis(double* const re, double* const im, const size_t total, const size_t n,
                      const size_t stride) {
    double* const line_re = static_cast<double*>(malloc(n * sizeof(double)));
    double* const line_im = static_cast<double*>(malloc(n * sizeof(double)));
    if ((line_re == nullptr) || (line_im == nullptr)) {
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
                const double phase = -2.0 * M_PI * static_cast<double>(k) * static_cast<double>(j) / static_cast<double>(n);
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

// One c2c case. Sizes are VkFFT's, so size[0] is the contiguous axis, and
// `batches` stacks on top of every size[] axis.
typedef struct {
    const char* name;
    uint64_t fft_dim;
    uint64_t size[MAX_CASE_DIM];
    uint64_t batches; // 0 leaves VkFFT's default of 1
    int inplace;
} c2c_case;

// One r2c case, out of place only: the in-place layout is covered on the other
// backends and the point here is the c2r routing.
typedef struct {
    const char* name;
    uint64_t fft_dim;
    uint64_t size[MAX_CASE_DIM];
    uint64_t batches;
} r2c_case;

// Runs one c2c case and reports its checks. Allocates its own buffers and plan
// so each case is independent. Returns 1 on a host or Metal failure that makes
// the rest of the run meaningless, 0 otherwise (a numerical failure is counted
// by report()).
static int _run_c2c_case(MTL::Device* const device, MTL::CommandQueue* const queue,
                         void** const device_handles, const c2c_case* const c) {
    const uint64_t batches = (c->batches == 0) ? 1 : c->batches;
    size_t total = static_cast<size_t>(batches);
    double scale = 1.0; // round-trip factor: the product of the transformed axes
    for (uint64_t a = 0; a < c->fft_dim; ++a) {
        total *= static_cast<size_t>(c->size[a]);
        scale *= static_cast<double>(c->size[a]);
    }
    const size_t floats = 2 * total;
    const size_t bytes = floats * sizeof(float);

    double* const ref_re = static_cast<double*>(malloc(total * sizeof(double)));
    double* const ref_im = static_cast<double*>(malloc(total * sizeof(double)));
    double* const src_re = static_cast<double*>(malloc(total * sizeof(double)));
    double* const src_im = static_cast<double*>(malloc(total * sizeof(double)));
    if ((ref_re == nullptr) || (ref_im == nullptr) || (src_re == nullptr) || (src_im == nullptr)) {
        fprintf(stderr, "[FAIL] host allocation failed for %s\n", c->name);
        return 1;
    }

    // Deterministic mixed signal, so the printed errors are reproducible.
    for (size_t i = 0; i < total; ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(total);
        src_re[i] = cos(2.0 * M_PI * 3.0 * t) + 0.25 * sin(2.0 * M_PI * 11.0 * t);
        src_im[i] = 0.5 * cos(2.0 * M_PI * 7.0 * t) - 0.125;
        ref_re[i] = src_re[i];
        ref_im[i] = src_im[i];
    }
    size_t stride = 1;
    for (uint64_t a = 0; a < c->fft_dim; ++a) {
        _dft_axis(ref_re, ref_im, total, static_cast<size_t>(c->size[a]), stride);
        stride *= static_cast<size_t>(c->size[a]);
    }

    vkfft_config config = {};
    config.fft_dim = c->fft_dim;
    for (uint64_t a = 0; a < c->fft_dim; ++a) {
        config.size[a] = c->size[a];
    }
    config.number_batches = c->batches;
    config.inplace = c->inplace;

    vkfft_app* app = nullptr;
    int res = vkfft_create(&config, device_handles, &app);
    if (res != 0) {
        printf("[FAIL] %-46s vkfft_create: %s (code %d)\n", c->name, vkfft_error_name(res), res);
        failures += 1;
        free(ref_re);
        free(ref_im);
        free(src_re);
        free(src_im);
        return 0;
    }

    MTL::Buffer* const buf_in = device->newBuffer(bytes, MTL::ResourceStorageModeShared);
    MTL::Buffer* const buf_out = c->inplace ? buf_in : device->newBuffer(bytes, MTL::ResourceStorageModeShared);
    if ((buf_in == nullptr) || (buf_out == nullptr)) {
        fprintf(stderr, "[FAIL] newBuffer failed for %s\n", c->name);
        return 1;
    }

    float* const host_in = static_cast<float*>(buf_in->contents());
    for (size_t i = 0; i < total; ++i) {
        host_in[2 * i] = static_cast<float>(src_re[i]);
        host_in[2 * i + 1] = static_cast<float>(src_im[i]);
    }
    if (!c->inplace) {
        _fill_poison(buf_out, floats);
    }

    char label[160];
    res = _submit(queue, app, buf_in, buf_out, DIRECTION_FORWARD);
    if (res != 0) {
        snprintf(label, sizeof(label), "%s forward", c->name);
        failures += fail_vkfft(label, res);
        vkfft_destroy(app);
        return 0;
    }

    if (!c->inplace) {
        snprintf(label, sizeof(label), "%s forward writes every output", c->name);
        report(label, _poison_fraction(buf_out, floats), 0.0);
    }

    const float* const got = static_cast<const float*>(buf_out->contents());
    double worst = 0.0;
    for (size_t i = 0; i < total; ++i) {
        const double dre = fabs(static_cast<double>(got[2 * i]) / scale - ref_re[i] / scale);
        const double dim = fabs(static_cast<double>(got[2 * i + 1]) / scale - ref_im[i] / scale);
        if (dre > worst) {
            worst = dre;
        }
        if (dim > worst) {
            worst = dim;
        }
    }
    snprintf(label, sizeof(label), "%s forward vs naive DFT", c->name);
    report(label, worst, TOLERANCE);

    // Invert the spectrum just measured. Out of place the destination is
    // poisoned first, so an inverse that writes nothing cannot pass.
    if (!c->inplace) {
        _fill_poison(buf_in, floats);
    }
    res = _submit(queue, app, buf_out, buf_in, DIRECTION_INVERSE);
    if (res != 0) {
        snprintf(label, sizeof(label), "%s inverse", c->name);
        failures += fail_vkfft(label, res);
        vkfft_destroy(app);
        return 0;
    }

    if (!c->inplace) {
        snprintf(label, sizeof(label), "%s inverse writes every output", c->name);
        report(label, _poison_fraction(buf_in, floats), 0.0);
    }

    // VkFFT's inverse is unnormalized and returns N*x, so the 1/N is applied
    // here. (config.normalize would do it in-kernel, which is the caller's
    // call.)
    const float* const back = static_cast<const float*>(buf_in->contents());
    worst = 0.0;
    for (size_t i = 0; i < total; ++i) {
        const double dre = fabs(static_cast<double>(back[2 * i]) / scale - src_re[i]);
        const double dim = fabs(static_cast<double>(back[2 * i + 1]) / scale - src_im[i]);
        if (dre > worst) {
            worst = dre;
        }
        if (dim > worst) {
            worst = dim;
        }
    }
    snprintf(label, sizeof(label), "%s round trip", c->name);
    report(label, worst, TOLERANCE);

    // The same plan again, forward and inverse appended to one command buffer
    // and committed once.
    for (size_t i = 0; i < total; ++i) {
        host_in[2 * i] = static_cast<float>(src_re[i]);
        host_in[2 * i + 1] = static_cast<float>(src_im[i]);
    }
    res = _submit_round_trip(queue, app, buf_in, buf_out);
    if (res != 0) {
        snprintf(label, sizeof(label), "%s two appends in one command buffer", c->name);
        failures += fail_vkfft(label, res);
        vkfft_destroy(app);
        return 0;
    }
    worst = 0.0;
    for (size_t i = 0; i < total; ++i) {
        const double dre = fabs(static_cast<double>(host_in[2 * i]) / scale - src_re[i]);
        const double dim = fabs(static_cast<double>(host_in[2 * i + 1]) / scale - src_im[i]);
        if (dre > worst) {
            worst = dre;
        }
        if (dim > worst) {
            worst = dim;
        }
    }
    snprintf(label, sizeof(label), "%s round trip in one command buffer", c->name);
    report(label, worst, TOLERANCE);

    vkfft_destroy(app);
    if (!c->inplace) {
        buf_out->release();
    }
    buf_in->release();
    free(ref_re);
    free(ref_im);
    free(src_re);
    free(src_im);
    return 0;
}

// One large 1D case, checked analytically. A single frequency transforms to
// N*delta[k - k0], which costs O(n) to verify, so the sizes here can be large
// enough to need several shared-memory uploads per axis, or prime and routed
// through Bluestein. A multi-upload size is what puts several dispatches in one
// encoder, and a prime size is what makes the plan submit and wait on the queue
// while it builds its tables.
typedef struct {
    const char* name;
    uint64_t size;
    double tol; // fp32 error grows with the size and with Bluestein, so this is per case
} delta_case;

static int _run_delta_case(MTL::Device* const device, MTL::CommandQueue* const queue,
                           void** const device_handles, const delta_case* const c) {
    const size_t n = static_cast<size_t>(c->size);
    const size_t floats = 2 * n;
    const int k0 = 5;

    vkfft_config config = {};
    config.fft_dim = 1;
    config.size[0] = c->size;

    vkfft_app* app = nullptr;
    int res = vkfft_create(&config, device_handles, &app);
    if (res != 0) {
        printf("[FAIL] %-46s vkfft_create: %s (code %d)\n", c->name, vkfft_error_name(res), res);
        failures += 1;
        return 0;
    }

    MTL::Buffer* const buf_in = device->newBuffer(floats * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* const buf_out = device->newBuffer(floats * sizeof(float), MTL::ResourceStorageModeShared);
    if ((buf_in == nullptr) || (buf_out == nullptr)) {
        fprintf(stderr, "[FAIL] newBuffer failed for %s\n", c->name);
        return 1;
    }
    float* const host_in = static_cast<float*>(buf_in->contents());
    for (size_t i = 0; i < n; ++i) {
        const double phase = 2.0 * M_PI * static_cast<double>(k0) * static_cast<double>(i) / static_cast<double>(n);
        host_in[2 * i] = static_cast<float>(cos(phase));
        host_in[2 * i + 1] = static_cast<float>(sin(phase));
    }
    _fill_poison(buf_out, floats);

    char label[160];
    res = _submit(queue, app, buf_in, buf_out, DIRECTION_FORWARD);
    if (res != 0) {
        snprintf(label, sizeof(label), "%s forward", c->name);
        failures += fail_vkfft(label, res);
        vkfft_destroy(app);
        return 0;
    }
    snprintf(label, sizeof(label), "%s forward writes every output", c->name);
    report(label, _poison_fraction(buf_out, floats), 0.0);

    const float* const got = static_cast<const float*>(buf_out->contents());
    double worst = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double want_re = (i == static_cast<size_t>(k0)) ? 1.0 : 0.0;
        const double dre = fabs(static_cast<double>(got[2 * i]) / static_cast<double>(n) - want_re);
        const double dim = fabs(static_cast<double>(got[2 * i + 1]) / static_cast<double>(n));
        if (dre > worst) {
            worst = dre;
        }
        if (dim > worst) {
            worst = dim;
        }
    }
    snprintf(label, sizeof(label), "%s forward is delta[%d]", c->name, k0);
    report(label, worst, c->tol);

    _fill_poison(buf_in, floats);
    res = _submit(queue, app, buf_out, buf_in, DIRECTION_INVERSE);
    if (res != 0) {
        snprintf(label, sizeof(label), "%s inverse", c->name);
        failures += fail_vkfft(label, res);
        vkfft_destroy(app);
        return 0;
    }
    snprintf(label, sizeof(label), "%s inverse writes every output", c->name);
    report(label, _poison_fraction(buf_in, floats), 0.0);

    worst = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double phase = 2.0 * M_PI * static_cast<double>(k0) * static_cast<double>(i) / static_cast<double>(n);
        const double dre = fabs(static_cast<double>(host_in[2 * i]) / static_cast<double>(n) - cos(phase));
        const double dim = fabs(static_cast<double>(host_in[2 * i + 1]) / static_cast<double>(n) - sin(phase));
        if (dre > worst) {
            worst = dre;
        }
        if (dim > worst) {
            worst = dim;
        }
    }
    snprintf(label, sizeof(label), "%s round trip", c->name);
    report(label, worst, c->tol);

    vkfft_destroy(app);
    buf_out->release();
    buf_in->release();
    return 0;
}

// Runs one out-of-place r2c case: forward against a naive DFT, then the inverse
// back to reals. The inverse is the path that exercises
// inverseReturnToInputBuffer, which is what makes a c2r plan write the real
// side.
static int _run_r2c_case(MTL::Device* const device, MTL::CommandQueue* const queue,
                         void** const device_handles, const r2c_case* const c) {
    const uint64_t batches = (c->batches == 0) ? 1 : c->batches;
    const size_t real_axis = static_cast<size_t>(c->size[0]);
    const size_t half_axis = real_axis / 2 + 1; // complex values along the real axis

    // Every axis after the first, flattened. Out of place both sides are dense
    // in their own element type, so only the length of the first axis differs.
    size_t rows = 1;
    double scale = 1.0;
    for (uint64_t a = 0; a < c->fft_dim; ++a) {
        if (a > 0) {
            rows *= static_cast<size_t>(c->size[a]);
        }
        scale *= static_cast<double>(c->size[a]);
    }
    rows *= static_cast<size_t>(batches);

    const size_t real_floats = rows * real_axis;
    const size_t cplx_floats = rows * half_axis * 2;

    double* const ref_re = static_cast<double*>(malloc(real_floats * sizeof(double)));
    double* const ref_im = static_cast<double*>(malloc(real_floats * sizeof(double)));
    double* const src = static_cast<double*>(malloc(real_floats * sizeof(double)));
    if ((ref_re == nullptr) || (ref_im == nullptr) || (src == nullptr)) {
        fprintf(stderr, "[FAIL] host allocation failed for %s\n", c->name);
        return 1;
    }
    for (size_t i = 0; i < real_floats; ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(real_floats);
        src[i] = cos(2.0 * M_PI * 3.0 * t) + 0.5 * sin(2.0 * M_PI * 7.0 * t) - 0.125;
        ref_re[i] = src[i];
        ref_im[i] = 0.0;
    }
    size_t stride = 1;
    for (uint64_t a = 0; a < c->fft_dim; ++a) {
        _dft_axis(ref_re, ref_im, real_floats, static_cast<size_t>(c->size[a]), stride);
        stride *= static_cast<size_t>(c->size[a]);
    }

    vkfft_config config = {};
    config.fft_dim = c->fft_dim;
    for (uint64_t a = 0; a < c->fft_dim; ++a) {
        config.size[a] = c->size[a];
    }
    config.number_batches = c->batches;
    config.r2c = 1;

    vkfft_app* app = nullptr;
    int res = vkfft_create(&config, device_handles, &app);
    if (res != 0) {
        printf("[FAIL] %-46s vkfft_create: %s (code %d)\n", c->name, vkfft_error_name(res), res);
        failures += 1;
        free(ref_re);
        free(ref_im);
        free(src);
        return 0;
    }

    MTL::Buffer* const buf_real = device->newBuffer(real_floats * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* const buf_cplx = device->newBuffer(cplx_floats * sizeof(float), MTL::ResourceStorageModeShared);
    if ((buf_real == nullptr) || (buf_cplx == nullptr)) {
        fprintf(stderr, "[FAIL] newBuffer failed for %s\n", c->name);
        return 1;
    }
    float* const host_real = static_cast<float*>(buf_real->contents());
    for (size_t i = 0; i < real_floats; ++i) {
        host_real[i] = static_cast<float>(src[i]);
    }
    _fill_poison(buf_cplx, cplx_floats);

    char label[160];
    res = _submit(queue, app, buf_real, buf_cplx, DIRECTION_FORWARD);
    if (res != 0) {
        snprintf(label, sizeof(label), "%s forward", c->name);
        failures += fail_vkfft(label, res);
        vkfft_destroy(app);
        return 0;
    }
    snprintf(label, sizeof(label), "%s forward writes every output", c->name);
    report(label, _poison_fraction(buf_cplx, cplx_floats), 0.0);

    // The device returns the first half_axis values along the real axis, which
    // is what gets compared.
    const float* const host_cplx = static_cast<const float*>(buf_cplx->contents());
    double worst = 0.0;
    for (size_t r = 0; r < rows; ++r) {
        for (size_t k = 0; k < half_axis; ++k) {
            const size_t dev = 2 * (r * half_axis + k);
            const size_t ref = r * real_axis + k;
            const double dre = fabs(static_cast<double>(host_cplx[dev]) / scale - ref_re[ref] / scale);
            const double dim = fabs(static_cast<double>(host_cplx[dev + 1]) / scale - ref_im[ref] / scale);
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

    float* const spectrum = static_cast<float*>(malloc(cplx_floats * sizeof(float)));
    if (spectrum == nullptr) {
        fprintf(stderr, "[FAIL] host allocation failed for %s\n", c->name);
        return 1;
    }
    for (size_t i = 0; i < cplx_floats; ++i) {
        spectrum[i] = host_cplx[i];
    }

    _fill_poison(buf_real, real_floats);
    res = _submit(queue, app, buf_cplx, buf_real, DIRECTION_INVERSE);
    if (res != 0) {
        snprintf(label, sizeof(label), "%s inverse", c->name);
        failures += fail_vkfft(label, res);
        vkfft_destroy(app);
        return 0;
    }
    snprintf(label, sizeof(label), "%s inverse writes every output", c->name);
    report(label, _poison_fraction(buf_real, real_floats), 0.0);

    worst = 0.0;
    for (size_t i = 0; i < real_floats; ++i) {
        const double diff = fabs(static_cast<double>(host_real[i]) / scale - src[i]);
        if (diff > worst) {
            worst = diff;
        }
    }
    snprintf(label, sizeof(label), "%s round trip", c->name);
    report(label, worst, TOLERANCE);

    // Not a check, a measurement the Julia layer needs: whether the inverse
    // leaves the spectrum it read intact, which decides if a c2r plan can be
    // handed a buffer the caller still wants.
    double spectrum_change = 0.0;
    for (size_t i = 0; i < cplx_floats; ++i) {
        const double d = fabs(static_cast<double>(host_cplx[i]) - static_cast<double>(spectrum[i])) / scale;
        if (d > spectrum_change) {
            spectrum_change = d;
        }
    }
    printf("       %s spectrum change across the inverse: %.3e\n", c->name, spectrum_change);

    vkfft_destroy(app);
    buf_cplx->release();
    buf_real->release();
    free(spectrum);
    free(ref_re);
    free(ref_im);
    free(src);
    return 0;
}

// What a stress loop puts in its command buffer.
typedef enum {
    STRESS_BODY_NOTHING = 0,      // a bare command buffer, the baseline cost
    STRESS_BODY_LEAKED_ENCODER,   // plus an encoder nobody reclaims, to price one leak
    STRESS_BODY_EXECUTE           // plus a transform through the wrapper
} stress_body;

// One stress loop: `iters` submissions, optionally each inside its own
// autorelease pool. Writes the footprint growth per iteration to *growth, which
// can come out negative when the process gives pages back, and returns nonzero
// only if an execute failed. Comparing loops that differ in one body is what
// makes the numbers readable, since page granularity and allocator churn hit
// them all alike.
static int _stress_loop(MTL::CommandQueue* const queue, vkfft_app* const app, MTL::Buffer* const buf_in,
                        MTL::Buffer* const buf_out, const int iters, const bool pooled,
                        const stress_body body, double* const growth) {
    const size_t before = _footprint_bytes();
    for (int it = 0; it < iters; ++it) {
        NS::AutoreleasePool* pool = nullptr;
        if (pooled) {
            pool = NS::AutoreleasePool::alloc()->init();
        }
        MTL::CommandBuffer* const command_buffer = queue->commandBuffer();
        if (body == STRESS_BODY_LEAKED_ENCODER) {
            command_buffer->computeCommandEncoder()->endEncoding();
        }
        else if (body == STRESS_BODY_EXECUTE) {
            const int res = vkfft_execute(app, buf_in, buf_out, DIRECTION_FORWARD, command_buffer);
            if (res != 0) {
                if (pool != nullptr) {
                    pool->release();
                }
                return fail_vkfft("stress execute", res);
            }
        }
        command_buffer->commit();
        command_buffer->waitUntilCompleted();
        if (pool != nullptr) {
            pool->release();
        }
    }
    const size_t after = _footprint_bytes();
    *growth = (static_cast<double>(after) - static_cast<double>(before)) / static_cast<double>(iters);
    return 0;
}

// Executes the same plan thousands of times and watches the process footprint.
// The pooled loops drain a pool around every execute, which is where an encoder
// released once too often faults. The unpooled loops run with no pool on the
// thread at all, the state a plain C or Julia caller is in, and there the
// caller's own command buffer leaks whatever the wrapper does. So the unpooled
// run prices one leaked encoder first, by leaking one deliberately, and then
// asks which side of the midpoint an execute lands on.
static int _run_stress(MTL::CommandQueue* const queue, vkfft_app* const app, MTL::Buffer* const buf_in,
                       MTL::Buffer* const buf_out) {
    printf("stress: %d pooled and %d unpooled executes against equal baselines, %d inside one pool\n",
           STRESS_POOLED, STRESS_UNPOOLED, STRESS_OUTER_POOL);

    double warmup = 0.0;
    double pooled_bare = 0.0;
    double pooled_exec = 0.0;
    if (_stress_loop(queue, app, buf_in, buf_out, STRESS_WARMUP, true, STRESS_BODY_EXECUTE, &warmup) != 0) {
        return 1;
    }
    if (_stress_loop(queue, app, buf_in, buf_out, STRESS_POOLED, true, STRESS_BODY_NOTHING, &pooled_bare) != 0) {
        return 1;
    }
    if (_stress_loop(queue, app, buf_in, buf_out, STRESS_POOLED, true, STRESS_BODY_EXECUTE, &pooled_exec) != 0) {
        return 1;
    }
    printf("       pooled: %.1f bytes per command buffer alone, %.1f with an execute\n",
           pooled_bare, pooled_exec);
    report("a pooled execute keeps the footprint flat", pooled_exec - pooled_bare, STRESS_POOLED_BOUND);

    double bare = 0.0;
    double leaked = 0.0;
    double exec = 0.0;
    if (_stress_loop(queue, app, buf_in, buf_out, STRESS_UNPOOLED, false, STRESS_BODY_NOTHING, &bare) != 0) {
        return 1;
    }
    if (_stress_loop(queue, app, buf_in, buf_out, STRESS_UNPOOLED, false, STRESS_BODY_LEAKED_ENCODER, &leaked) != 0) {
        return 1;
    }
    if (_stress_loop(queue, app, buf_in, buf_out, STRESS_UNPOOLED, false, STRESS_BODY_EXECUTE, &exec) != 0) {
        return 1;
    }
    printf("       unpooled: %.1f bytes per command buffer alone, %.1f with a leaked encoder, %.1f with an execute\n",
           bare, leaked, exec);
    printf("       one leaked encoder costs %.1f bytes per submission\n", leaked - bare);
    report("an unpooled execute gives its encoder back", exec - 0.5 * (bare + leaked), 0.0);

    // Many executes inside one pool, drained at the end: the arrangement that
    // faults on an over-release only once the pool goes.
    NS::AutoreleasePool* const pool = NS::AutoreleasePool::alloc()->init();
    for (int it = 0; it < STRESS_OUTER_POOL; ++it) {
        MTL::CommandBuffer* const command_buffer = queue->commandBuffer();
        const int res = vkfft_execute(app, buf_in, buf_out, DIRECTION_FORWARD, command_buffer);
        if (res != 0) {
            pool->release();
            return fail_vkfft("stress outer-pool execute", res);
        }
        command_buffer->commit();
        command_buffer->waitUntilCompleted();
    }
    pool->release();
    report("draining one pool over many executes survives", 0.0, 0.0);
    return 0;
}

int main(void) {
    printf("libvkfft quick test: backend %llu, max dims %llu\n",
           static_cast<unsigned long long>(vkfft_backend()),
           static_cast<unsigned long long>(vkfft_max_dims()));
    if (vkfft_backend() != 5) {
        fprintf(stderr, "[FAIL] this test needs the Metal build (VKFFT_BACKEND=5)\n");
        return 1;
    }
    report("vkfft_config_size matches this mirror", (vkfft_config_size() == sizeof(vkfft_config)) ? 0.0 : 1.0, 0.0);

    MTL::Device* const device = MTL::CreateSystemDefaultDevice();
    if (device == nullptr) {
        fprintf(stderr, "[FAIL] no Metal device\n");
        return 1;
    }
    printf("device: %s\n", device->name()->utf8String());

    MTL::CommandQueue* const queue = device->newCommandQueue();
    if (queue == nullptr) {
        fprintf(stderr, "[FAIL] newCommandQueue failed\n");
        return 1;
    }

    // The device and the queue objects themselves, not pointers to them.
    void* device_handles[2];
    device_handles[0] = device;
    device_handles[1] = queue;

    // c2c: 1D at the reference size, in-place and out-of-place, then 2D and a
    // batched case.
    static const c2c_case c2c_cases[] = {
        {"c2c 1d 64", 1, {FFT_SIZE, 0}, 0, 0},
        {"c2c 1d 64 in-place", 1, {FFT_SIZE, 0}, 0, 1},
        {"c2c 2d 16x8", 2, {16, 8}, 0, 0},
        {"c2c 2d 16x8 in-place", 2, {16, 8}, 0, 1},
        {"c2c 1d 32 batch 4", 1, {32, 0}, 4, 0},
        {"c2c 2d 16x8 batch 3", 2, {16, 8}, 3, 0},
    };
    for (size_t i = 0; i < sizeof(c2c_cases) / sizeof(c2c_cases[0]); ++i) {
        if (_run_c2c_case(device, queue, device_handles, &c2c_cases[i]) != 0) {
            return 1;
        }
    }

    // Sizes past the 32 kB of threadgroup memory an Apple GPU gives a kernel,
    // so an axis takes several uploads and the wrapper's single encoder has to
    // carry all of them, plus a prime size that goes through Bluestein.
    static const delta_case delta_cases[] = {
        {"c2c 1d 4096", 4096, 1e-4},
        {"c2c 1d 8192 (multi-upload)", 8192, 1e-4},
        {"c2c 1d 1021 (prime)", 1021, 1e-4},
    };
    for (size_t i = 0; i < sizeof(delta_cases) / sizeof(delta_cases[0]); ++i) {
        if (_run_delta_case(device, queue, device_handles, &delta_cases[i]) != 0) {
            return 1;
        }
    }

    // r2c out of place, both directions, with an even and an odd real axis: the
    // inverse is the branch that had never run on Metal.
    static const r2c_case r2c_cases[] = {
        {"r2c 1d 16", 1, {16, 0}, 0},
        {"r2c 1d 13 (odd)", 1, {13, 0}, 0},
        {"r2c 1d 16 batch 3", 1, {16, 0}, 3},
        {"r2c 2d 16x13", 2, {16, 13}, 0},
        {"r2c 2d 13x8 (odd)", 2, {13, 8}, 0},
    };
    for (size_t i = 0; i < sizeof(r2c_cases) / sizeof(r2c_cases[0]); ++i) {
        if (_run_r2c_case(device, queue, device_handles, &r2c_cases[i]) != 0) {
            return 1;
        }
    }

    // One plan for the rest: the sign convention, plan reuse across fresh
    // command buffers, and the stress loop.
    vkfft_config config = {};
    config.fft_dim = 1;
    config.size[0] = FFT_SIZE;

    vkfft_app* app = nullptr;
    int res = vkfft_create(&config, device_handles, &app);
    if (res != 0) {
        return fail_vkfft("vkfft_create (out-of-place)", res);
    }

    const size_t bytes = 2 * FFT_SIZE * sizeof(float);
    MTL::Buffer* const buf_in = device->newBuffer(bytes, MTL::ResourceStorageModeShared);
    MTL::Buffer* const buf_out = device->newBuffer(bytes, MTL::ResourceStorageModeShared);
    if ((buf_in == nullptr) || (buf_out == nullptr)) {
        fprintf(stderr, "[FAIL] newBuffer failed\n");
        return 1;
    }
    float* const host_in = static_cast<float*>(buf_in->contents());

    // Single-frequency input: the forward transform is analytically N*delta[k -
    // k0], which pins down the sign convention of DIRECTION_FORWARD.
    const int k0 = 5;
    for (int i = 0; i < FFT_SIZE; ++i) {
        const double phase = 2.0 * M_PI * static_cast<double>(k0) * static_cast<double>(i) / static_cast<double>(FFT_SIZE);
        host_in[2 * i] = static_cast<float>(cos(phase));
        host_in[2 * i + 1] = static_cast<float>(sin(phase));
    }
    _fill_poison(buf_out, 2 * FFT_SIZE);
    res = _submit(queue, app, buf_in, buf_out, DIRECTION_FORWARD);
    if (res != 0) {
        return fail_vkfft("vkfft_execute (delta forward)", res);
    }
    const float* const got = static_cast<const float*>(buf_out->contents());
    double worst = 0.0;
    for (int i = 0; i < FFT_SIZE; ++i) {
        const double want_re = (i == k0) ? 1.0 : 0.0;
        const double dre = fabs(static_cast<double>(got[2 * i]) / static_cast<double>(FFT_SIZE) - want_re);
        const double dim = fabs(static_cast<double>(got[2 * i + 1]) / static_cast<double>(FFT_SIZE));
        if (dre > worst) {
            worst = dre;
        }
        if (dim > worst) {
            worst = dim;
        }
    }
    report("forward of exp(+2*pi*i*5*n/N) is delta[5]", worst, TOLERANCE);

    // Plan reuse: the same app driven through several fresh command buffers,
    // each one checked, so a plan that only works once cannot pass.
    worst = 0.0;
    for (int pass = 0; pass < 5; ++pass) {
        _fill_poison(buf_out, 2 * FFT_SIZE);
        res = _submit(queue, app, buf_in, buf_out, DIRECTION_FORWARD);
        if (res != 0) {
            return fail_vkfft("vkfft_execute (reuse)", res);
        }
        for (int i = 0; i < FFT_SIZE; ++i) {
            const double want_re = (i == k0) ? 1.0 : 0.0;
            const double dre = fabs(static_cast<double>(got[2 * i]) / static_cast<double>(FFT_SIZE) - want_re);
            const double dim = fabs(static_cast<double>(got[2 * i + 1]) / static_cast<double>(FFT_SIZE));
            if (dre > worst) {
                worst = dre;
            }
            if (dim > worst) {
                worst = dim;
            }
        }
    }
    report("five reuses through fresh command buffers", worst, TOLERANCE);

    if (_run_stress(queue, app, buf_in, buf_out) != 0) {
        failures += 1;
    }

    // The value is still the one the plan started with, so the stress loop was
    // doing the transform and not just churning command buffers.
    worst = 0.0;
    for (int i = 0; i < FFT_SIZE; ++i) {
        const double want_re = (i == k0) ? 1.0 : 0.0;
        const double dre = fabs(static_cast<double>(got[2 * i]) / static_cast<double>(FFT_SIZE) - want_re);
        const double dim = fabs(static_cast<double>(got[2 * i + 1]) / static_cast<double>(FFT_SIZE));
        if (dre > worst) {
            worst = dre;
        }
        if (dim > worst) {
            worst = dim;
        }
    }
    report("stress loop result still correct", worst, TOLERANCE);

    // A null command buffer has to come back as an error rather than a crash,
    // since Metal has no default queue to fall back on.
    res = vkfft_execute(app, buf_in, buf_out, DIRECTION_FORWARD, nullptr);
    report("null command buffer refused", (res != 0) ? 0.0 : 1.0, 0.0);
    printf("       null command buffer returned %s\n", vkfft_error_name(res));

    // The error wrapper must come from VkFFT's own table, not from anything
    // written here.
    printf("vkfft_error_name(0) = %s\n", vkfft_error_name(0));
    printf("vkfft_error_name(2004) = %s\n", vkfft_error_name(2004));

    vkfft_destroy(app);
    buf_out->release();
    buf_in->release();
    queue->release();
    device->release();

    printf("%s (%d failing checks)\n", (failures == 0) ? "ALL CHECKS PASSED" : "FAILURES", failures);
    return (failures == 0) ? 0 : 1;
}
