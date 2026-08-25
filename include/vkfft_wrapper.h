// libvkfft: a backend-generic C wrapper for VkFFT
// (https://github.com/DTolm/VkFFT).
//
// VkFFT picks its backend at compile time, so one shared library is built per
// backend. This header is the same for all of them and the exported symbols are
// identical. The only thing that changes is the meaning of the opaque handles
// passed in and out. The wrapper copies vkfft_config into VkFFTConfiguration,
// keeps the buffer/queue slots VkFFT dereferences alive, and forwards result
// codes. Every parameter choice (sizes, batching, region mapping, direction,
// precision choice, synchronization) is made by the caller.

#ifndef VKFFT_WRAPPER_H
#define VKFFT_WRAPPER_H

#include <stdint.h>

// Must match the VKFFT_MAX_FFT_DIMENSIONS this library was compiled with (12 in
// every official build). Callers must not hardcode it and instead should read
// vkfft_max_dims() at load time and compare against the value baked into
// whatever mirror of vkfft_config they use.
#ifndef VKFFT_MAX_FFT_DIMENSIONS
#define VKFFT_MAX_FFT_DIMENSIONS 12
#endif

// Returned when the wrapper rejects an argument before VkFFT ever sees it (null
// pointer, or a precision outside 0..3). Every other code is a VkFFTResult
// passed through unchanged.
#define VKFFT_WRAPPER_ERROR_INVALID_ARGUMENT (-1)

#ifdef __cplusplus
extern "C" {
#endif

typedef struct vkfft_app vkfft_app;

// POD mirror of the subset of VkFFTConfiguration this wrapper exposes. Field
// order and types are part of the ABI: callers mirror this struct verbatim.
// Zero means "VkFFT's default" for every field, so a zeroed struct plus fft_dim
// and size is a valid c2c plan.
typedef struct {
    uint64_t fft_dim;                                 // VkFFT FFTdim
    uint64_t size[VKFFT_MAX_FFT_DIMENSIONS];          // all slots are read (unused ones must be 0)
    uint64_t omit[VKFFT_MAX_FFT_DIMENSIONS];          // VkFFT omitDimension: 1 = no FFT along this axis
    uint64_t number_batches;                          // 0 leaves VkFFT's default of 1
    int precision;                                    // 0=fp32, 1=fp64, 2=fp16, 3=quad (double-double)
    int r2c;                                          // 1 = R2C forward / C2R inverse
    int dct;                                          // DCT type 1..4, 0 = off
    int dst;                                          // DST type 1..4, 0 = off
    int inplace;                                      // 1 = one buffer, 0 = read `in` and write `out`
    int normalize;                                    // 1 = apply 1/N in-kernel on the inverse
    int make_forward_only;                            // skip inverse kernel generation
    int make_inverse_only;                            // skip forward kernel generation
    uint64_t zeropad_left[VKFFT_MAX_FFT_DIMENSIONS];  // VkFFT fft_zeropad_left
    uint64_t zeropad_right[VKFFT_MAX_FFT_DIMENSIONS]; // VkFFT fft_zeropad_right
    int perform_zeropad[VKFFT_MAX_FFT_DIMENSIONS];    // VkFFT performZeropadding, per axis
    uint64_t coalesced_memory;                        // tuning, in bytes, 0 = auto
    uint64_t aim_threads;                             // tuning, threads per block, 0 = auto
    uint64_t num_shared_banks;                        // tuning, 0 = auto
} vkfft_config;

// Creates a VkFFT application. Returns VKFFT_SUCCESS and writes *out on
// success. On failure, it returns the VkFFTResult (or
// VKFFT_WRAPPER_ERROR_INVALID_ARGUMENT) and sets *out = NULL. The wrapper
// copies the values behind device_handles, so the array and the variables it
// points at need not outlive this call. The device, context and queue objects
// themselves must outlive the app.
//
// device_handles is an array of opaque pointers whose length and meaning depend
// on the backend this library was built for:
//   CUDA   (VKFFT_BACKEND 1): { CUdevice* }
//   OpenCL (VKFFT_BACKEND 3): { cl_platform_id*, cl_device_id*, cl_context*,
//                               cl_command_queue* }
//   Metal  (VKFFT_BACKEND 5): { MTLDevice*, MTLCommandQueue* }
// The OpenCL command queue given here is only a default. vkfft_execute
// overrides it per call. On Metal the two entries are the device and queue
// objects themselves, not pointers to them.
int vkfft_create(const vkfft_config* config, void** device_handles, vkfft_app** out);

// Runs one transform. Buffer handles and the stream/queue are set on every
// call, so a plan can be reused across buffers and across streams. `in` is the
// input of this call and `out` is its output, whatever the direction.
//
// `in`, `out` are device buffer handles:
//   CUDA:   device pointers (the values cuMemAlloc / CUDA.jl hand out)
//   OpenCL: cl_mem handles (itself a pointer type, so it fits void* directly)
//   Metal:  MTLBuffer objects
// `stream` is the per-call submission handle:
//   CUDA:   cudaStream_t (may be NULL, which selects the legacy default stream)
//   OpenCL: cl_command_queue, required
//   Metal:  MTLCommandBuffer, required. The wrapper opens and ends a compute
//           encoder on it but never commits, so the caller controls submission.
// `direction` is forwarded to VkFFTAppend unchanged: -1 (or 0) forward, 1
// inverse.
//
// For an in-place plan VkFFT works out of a single buffer: `out` is used and
// `in` is ignored, so callers should pass the same handle twice.
//
// An out-of-place r2c plan has a real side and a complex side, and which handle
// is which follows the direction. Forward takes `in` as the real side and
// writes the spectrum to `out`. The inverse takes `in` as the spectrum and
// writes reals to `out`. Per batch, the real side holds
// prod(size[0..fft_dim-1]) reals and the complex side holds
// (size[0]/2 + 1) * prod(size[1..fft_dim-1]) complex values. Both are dense in
// their own element type, so the real side carries no row padding (unlike the
// in-place layout, where rows are padded to 2 * (size[0]/2 + 1) reals). The
// inverse uses the spectrum as scratch space, so treat `in` as clobbered. It
// comes back untouched only when a single axis is transformed.
int vkfft_execute(vkfft_app* app, void* in, void* out, int direction, void* stream);

// Frees the application. Does nothing on NULL. Device buffers are the caller's.
void vkfft_destroy(vkfft_app* app);

// Name of a result code, from VkFFT's own getVkFFTErrorString. Never NULL.
const char* vkfft_error_name(int res);

// The VKFFT_MAX_FFT_DIMENSIONS this library was built with, i.e., the array
// length in vkfft_config.
uint64_t vkfft_max_dims(void);

// The VKFFT_BACKEND baked into this library: 1 CUDA, 3 OpenCL, 5 Metal.
uint64_t vkfft_backend(void);

#ifdef __cplusplus
}
#endif

#endif // VKFFT_WRAPPER_H
