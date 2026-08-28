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
// pointer, a precision outside 0..3, a kernel set on a plan that does no
// convolution, or a binary blob shorter than its own header says). Every code
// other than the two below is a VkFFTResult passed through unchanged.
#define VKFFT_WRAPPER_ERROR_INVALID_ARGUMENT (-1)

// Returned when the request is well formed but the backend this library was
// built for cannot do it at all. The only such case is kernel binary
// save/load on Metal, which VkFFT compiles out.
#define VKFFT_WRAPPER_ERROR_UNSUPPORTED (-2)

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
    // The struct only ever grows at this end, so a caller's mirror keeps every
    // offset above valid. A mirror that stops short of the fields below is
    // still smaller than what the wrapper reads, so it has to cover all of
    // them.
    uint64_t coordinate_features;                     // VkFFT coordinateFeatures, 0 leaves VkFFT's default of 1
    uint64_t number_kernels;                          // VkFFT numberKernels, 0 leaves VkFFT's default of 1
    int perform_convolution;                          // VkFFT performConvolution, 1 = fft, multiply, inverse in one plan
    int kernel_convolution;                           // VkFFT kernelConvolution, 1 = this plan only transforms a kernel
    int conjugate_convolution;                        // VkFFT conjugateConvolution, 1 = conjugate the transformed input
    int save_to_string;                               // VkFFT saveApplicationToString, 1 = keep the compiled binaries for vkfft_save
} vkfft_config;

// Fused convolution needs two applications, and the caller builds both.
//
// The first has kernel_convolution = 1 and performs an ordinary forward
// transform of the kernel, in whatever layout the second one will read. It
// exists as its own plan because VkFFT disables a few internal reorderings for
// it so that its output matches what the convolution step expects. Run it once
// with vkfft_execute in the forward direction and keep the buffer it wrote.
//
// The second has perform_convolution = 1 and every layout field identical to
// the first, and it is pointed at that buffer with vkfft_set_kernel. Executing
// it in the forward direction runs the whole pipeline: forward transform of the
// input, elementwise multiply against the kernel, inverse transform back. The
// inverse half of a convolution plan always applies its own 1/N, whatever
// `normalize` says, so the result is the plain circular convolution.
//
// conjugate_convolution = 1 conjugates the transformed input before the
// multiply, which turns the pipeline into a circular cross-correlation:
// out[m] = sum_n conj(in[n]) * kernel[n + m]. VkFFT documents a value of 2 as
// conjugating the kernel instead, but no version of VkFFT vendored here
// generates code for it, so 2 behaves as 0.
//
// coordinate_features stacks independent components inside one plan: the buffer
// holds coordinate_features contiguous copies of the whole transform layout,
// component c multiplied by component c of the kernel. number_kernels stacks
// kernels the other way: one input is convolved against number_kernels of them
// and the output buffer holds number_kernels copies of the layout. The kernel
// buffer then holds number_kernels * coordinate_features copies, which is what
// a kernel plan with number_batches = number_kernels writes.
//
// perform_convolution excludes make_forward_only and make_inverse_only. The
// pipeline is forward, multiply, inverse, so a convolution plan needs both
// halves even though the caller only ever executes it in the forward
// direction. Building one half leaves the other plan unallocated and VkFFT
// dereferences it during the append rather than reporting anything, so
// vkfft_create refuses the combination with
// VKFFT_WRAPPER_ERROR_INVALID_ARGUMENT. A kernel_convolution plan is an
// ordinary forward transform and make_forward_only is the right thing to set
// on it.
//
// VkFFT refuses a convolution plan that also omits an axis.
//
// Two shapes do not work. Both are upstream limitations rather than wrapper
// ones, and VkFFT reports neither as an error.
//
// A convolution plan with fft_dim 1 over a length VkFFT transforms in a single
// upload fails to compile: the generated code opens a bounds check per
// register and never closes it. Writing the same transform as fft_dim 2 with a
// trailing axis of length 1 is equivalent, moves the multiply onto the strided
// code path, and works, so that is the shape to use for a 1D convolution.
// Lengths large enough to need several uploads (roughly 8192 and up) take the
// strided path on their own and compile as fft_dim 1.
//
// number_kernels above 1 needs a genuinely multi-axis transform. Combined with
// the trailing length-1 axis of that 1D workaround it runs without error and
// writes zeros over the whole output, so a batched-kernel convolution needs a
// real second axis. number_kernels = 1 is unaffected, and so is
// coordinate_features, which works on either shape.

// Points a convolution plan at the buffer holding its pre-transformed kernel.
// Returns VKFFT_SUCCESS, or VKFFT_WRAPPER_ERROR_INVALID_ARGUMENT for a null
// argument or an app that was not created with perform_convolution.
//
// `kernel` is a device buffer handle of the same kind vkfft_execute takes.
// Like the buffer slots, the wrapper owns the storage and VkFFT reads it at
// every dispatch, so the handle set here persists across vkfft_execute calls
// until the next vkfft_set_kernel. It has to be set at least once before the
// first execute, and vkfft_execute returns VKFFT_ERROR_EMPTY_kernel if it was
// not. Nothing about the buffer is copied, so it must stay alive and hold the
// transformed kernel for as long as the plan is used.
int vkfft_set_kernel(vkfft_app* app, void* kernel);

// Hands back the compiled kernel binaries of a plan created with
// save_to_string = 1, for later reuse through vkfft_create_loaded. Returns
// VKFFT_SUCCESS and writes *data and *size on success.
//
// The bytes belong to VkFFT, not the caller. They are allocated during
// vkfft_create and freed by vkfft_destroy, and nothing else in the lifetime of
// the app touches them, so the pointer is valid until vkfft_destroy and must
// not be freed. A caller that wants them to outlive the app copies `size`
// bytes out. The blob carries its own length in its first eight bytes, so
// *size is the whole of it and the same value must come back to
// vkfft_create_loaded.
//
// Returns VKFFT_ERROR_EMPTY_applicationString if the plan was not created with
// save_to_string, and VKFFT_WRAPPER_ERROR_UNSUPPORTED on Metal, where VkFFT
// compiles the save path out.
int vkfft_save(vkfft_app* app, const char** data, uint64_t* size);

// vkfft_create with the compiled binaries supplied instead of generated. Same
// arguments and same result codes, plus `data` and `size` from an earlier
// vkfft_save. Skipping the compile is the entire point, so expect this to
// return far faster than vkfft_create on the same config.
//
// `config` must describe the same plan the binaries were saved from, down to
// the tuning fields, since VkFFT lays the blob out in the order that config
// generates kernels in and reads it back positionally with no cross-check. It
// must have save_to_string = 0: VkFFT rejects loading and saving at once. The
// binaries are device-specific and VkFFT-version-specific, and neither is
// checked here or in VkFFT, so a stale or foreign blob fails in whatever way
// the driver fails to load it.
//
// The wrapper checks `size` against the length the blob claims and returns
// VKFFT_WRAPPER_ERROR_INVALID_ARGUMENT if the buffer is too short, since VkFFT
// itself reads the blob purely by that internal length. `data` is read during
// this call only and is not retained, so it can be freed as soon as the call
// returns. On Metal this returns VKFFT_WRAPPER_ERROR_UNSUPPORTED.
int vkfft_create_loaded(const vkfft_config* config, void** device_handles, const char* data, uint64_t size, vkfft_app** out);

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
//
// Metal callers have two extra obligations here. VkFFT builds its lookup and
// Bluestein tables during this call by submitting to the queue and waiting for
// it, so vkfft_create synchronizes and the queue has to be able to run work.
// And VkFFT's Metal plan builder over-releases the strings it compiles kernels
// from, so this call must not run inside an autorelease pool that is drained
// afterwards. Drain any pool the thread holds before calling, or call from a
// thread that has none. vkfft_execute has no such restriction.
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
// The Metal call sequence is: the caller makes a command buffer from its queue,
// calls vkfft_execute once or several times on that one command buffer, then
// commits and waits. Each call opens its own compute encoder, appends every
// dispatch of the transform to it, and ends it, so the encoders run in
// submission order and consecutive transforms see each other's results. The
// encoder belongs to the wrapper and is gone when the call returns. The command
// buffer belongs to the caller: it comes back autoreleased from the queue, so
// the caller must hold an autorelease pool rather than release it by hand.
//
// A buffer crosses as the whole MTLBuffer, and the launch parameters carry no
// offset the wrapper can set, so `in` and `out` must be buffers whose first
// element is the first element of the transform. An array that is a view at a
// nonzero offset into its buffer has to be rejected or copied by the caller,
// exactly as on OpenCL.
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

// sizeof(vkfft_config) as this library reads it. A caller compares it against
// its own mirror to catch a mirror that stops short of the fields the wrapper
// reads, which no other check can see: the struct only grows at the end, so a
// short mirror keeps every offset valid and fails silently.
uint64_t vkfft_config_size(void);

// The CUDA_TOOLKIT_ROOT_DIR this library was compiled with. Never NULL, and
// empty when nothing was baked in, which is every OpenCL and Metal build and a
// CUDA build whose toolkit root was left unset. The returned pointer is a
// static string owned by the library, so the caller must not free it and it
// stays valid for the lifetime of the process.
//
// An empty value means half precision cannot compile on this build. VkFFT
// generates its half-precision CUDA kernels with an
// #include <ROOT/include/cuda_fp16.h> assembled from this macro, and it hands
// nvrtc no header list and no -I of its own, so an empty root leaves the
// generated source including </include/cuda_fp16.h> and the compile fails after
// VkFFT has dumped the whole kernel to stdout. Callers check this before
// planning a half-precision transform on CUDA rather than after.
//
// Additive: this symbol changes no struct, so vkfft_config_size and the mirror
// check built on it are unaffected.
const char* vkfft_cuda_toolkit_root(void);

#ifdef __cplusplus
}
#endif

#endif // VKFFT_WRAPPER_H
