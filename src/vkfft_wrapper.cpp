// One translation unit for every backend. vkfft_wrapper.h is included first so
// that its VKFFT_MAX_FFT_DIMENSIONS default is in place before vkFFT.h picks
// its own.

#include "vkfft_wrapper.h"

#include <stdlib.h>
#include <string.h>

#include "vkFFT.h"

// vkFFT.h pulls in CL/cl_platform.h, which includes altivec.h on any target
// that has AltiVec. GCC's altivec.h defines bool, vector and pixel as macros
// for its own vector keywords, which turns the plain bool declarations in
// vkfft_execute into vector types and breaks the powerpc64le build. That
// header is written to be undone this way. All three macros go out together
// because they come in together, and leaving one behind would catch out the
// next declaration written here.
#if defined(__VEC__)
#undef bool
#undef vector
#undef pixel
#endif

#if (VKFFT_BACKEND == 1)
typedef void* buffer_handle; // CUDA device pointer
typedef cudaStream_t stream_handle;
#elif (VKFFT_BACKEND == 3)
typedef cl_mem buffer_handle;
typedef cl_command_queue stream_handle;
#elif (VKFFT_BACKEND == 5)
typedef MTL::Buffer* buffer_handle;
typedef MTL::CommandBuffer* stream_handle;
#else
#error "libvkfft supports VKFFT_BACKEND 1 (CUDA), 3 (OpenCL) and 5 (Metal) only"
#endif

// VkFFT stores the *addresses* of the buffer and stream handles in its
// configuration and dereferences them at every dispatch, so the wrapper owns
// the slots and rewrites their contents per call. The CUDA and OpenCL device
// handles are copied in for the same reason: the caller's storage may be a
// temporary. (Metal is different: VkFFT holds the device and queue objects
// themselves, so the caller keeps owning them.)
struct vkfft_app {
    VkFFTApplication app; // must be first: vkfft_create zeroes it
    buffer_handle buffer;
    buffer_handle input_buffer;
    buffer_handle kernel_buffer; // convolution only, rewritten by vkfft_set_kernel
#if (VKFFT_BACKEND == 1)
    CUdevice device;
    stream_handle stream;
#elif (VKFFT_BACKEND == 3)
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    stream_handle queue;
#endif
};

// The whole of vkfft_create and vkfft_create_loaded. `data` is null for the
// former and the saved binaries for the latter, which is the only difference
// between them.
static int create_app(const vkfft_config* const config, void** const device_handles, const char* const data, const uint64_t size, vkfft_app** const out) {
    if ((config == nullptr) || (device_handles == nullptr) || (out == nullptr)) {
        return VKFFT_WRAPPER_ERROR_INVALID_ARGUMENT;
    }
    *out = nullptr;

#if (VKFFT_BACKEND == 5)
    // VkFFT compiles both halves of the binary cache out on Metal: the save
    // branch of its compile step is empty and it never copies the load pointer
    // into its own configuration. Requesting either would fail obscurely (a
    // load reports an empty application string even though one was given, a
    // save silently produces nothing and leaks the buffer it allocated for it),
    // so both are refused here instead.
    if ((config->save_to_string != 0) || (data != nullptr)) {
        return VKFFT_WRAPPER_ERROR_UNSUPPORTED;
    }
#endif

    // A convolution plan runs a forward transform, the kernel multiply, then
    // an inverse transform, so it needs both halves even though it is only
    // ever executed in the forward direction. Asking for one half leaves the
    // other plan unallocated and VkFFTAppend dereferences it, which is a crash
    // rather than a result code, so the combination is refused here.
    if ((config->perform_convolution != 0) && ((config->make_forward_only != 0) || (config->make_inverse_only != 0))) {
        return VKFFT_WRAPPER_ERROR_INVALID_ARGUMENT;
    }

    // VkFFT reads the blob positionally and takes its length from the first
    // eight bytes rather than from anything the caller passes, so a truncated
    // buffer would be read past the end. This is the one thing about it the
    // wrapper can check.
    if (data != nullptr) {
        uint64_t claimed = 0;
        if (size < 5 * sizeof(uint64_t)) { // 5: VkFFT's blob opens with a five-word header (total size in word 0, the Rader offset in word 2) that it reads before any other check.
            return VKFFT_WRAPPER_ERROR_INVALID_ARGUMENT;
        }
        memcpy(&claimed, data, sizeof(uint64_t));
        if (claimed > size) {
            return VKFFT_WRAPPER_ERROR_INVALID_ARGUMENT;
        }
    }

    // initializeVkFFT memcmps the whole VkFFTApplication against zero,
    // padding included, so calloc rather than new as it is the only
    // allocation that guarantees every byte.
    vkfft_app* const self = static_cast<vkfft_app*>(calloc(1, sizeof(vkfft_app)));
    if (self == nullptr) {
        return VKFFT_ERROR_MALLOC_FAILED;
    }

    VkFFTConfiguration vk = {};

    vk.FFTdim = config->fft_dim;
    for (uint64_t i = 0; i < VKFFT_MAX_FFT_DIMENSIONS; ++i) {
        vk.size[i] = config->size[i];
        vk.omitDimension[i] = config->omit[i];
        vk.performZeropadding[i] = static_cast<pfUINT>(config->perform_zeropad[i]);
        vk.fft_zeropad_left[i] = config->zeropad_left[i];
        vk.fft_zeropad_right[i] = config->zeropad_right[i];
    }
    vk.numberBatches = config->number_batches; // 0 stays 0 and VkFFT substitutes its default of 1

    switch (config->precision) {
        case 0: break; // fp32 is VkFFT's default
        case 1: vk.doublePrecision = 1; break;
        case 2: vk.halfPrecision = 1; break;
        case 3: vk.quadDoubleDoublePrecision = 1; break;
        default:
            free(self);
            return VKFFT_WRAPPER_ERROR_INVALID_ARGUMENT;
    }

    vk.performR2C = static_cast<pfUINT>(config->r2c);
    vk.performDCT = static_cast<pfUINT>(config->dct);
    vk.performDST = static_cast<pfUINT>(config->dst);
    vk.normalize = static_cast<pfUINT>(config->normalize);
    vk.makeForwardPlanOnly = static_cast<pfUINT>(config->make_forward_only);
    vk.makeInversePlanOnly = static_cast<pfUINT>(config->make_inverse_only);

    vk.coalescedMemory = config->coalesced_memory;
    vk.aimThreads = config->aim_threads;
    vk.numSharedBanks = config->num_shared_banks;

    // Convolution. VkFFT reads coordinateFeatures on every plan but the rest
    // only when performConvolution is set, and copying them unconditionally
    // matches that: a kernel plan takes whatever the caller gives it and the
    // ignored fields stay ignored.
    vk.coordinateFeatures = config->coordinate_features;
    vk.numberKernels = config->number_kernels;
    vk.performConvolution = static_cast<pfUINT>(config->perform_convolution);
    vk.kernelConvolution = static_cast<pfUINT>(config->kernel_convolution);
    vk.conjugateConvolution = static_cast<pfUINT>(config->conjugate_convolution);
    if (config->perform_convolution != 0) {
        // Same arrangement as the buffer slots: VkFFT keeps the address and
        // reads through it at every dispatch, so vkfft_set_kernel only has to
        // write the slot. kernelSize is left null for the same reason
        // bufferSize is, and kernelNum stays at VkFFT's default of one.
        vk.kernel = &self->kernel_buffer;
    }

    // Binary cache. Both flags have to be in place before initializeVkFFT
    // since that is where kernels are compiled and where the blob is
    // assembled. VkFFT rejects the two together itself.
    vk.saveApplicationToString = static_cast<pfUINT>(config->save_to_string);
    if (data != nullptr) {
        vk.loadApplicationFromString = 1;
        vk.loadApplicationString = const_cast<char*>(data); // read during initializeVkFFT and not retained
    }

    // Out-of-place is expressed as isInputFormatted: VkFFT reads the first
    // axis from inputBuffer and leaves the result in buffer. Setting
    // isOutputFormatted too would make VkFFT want a third, scratch
    // allocation, which the wrapper has no business owning.
    vk.buffer = &self->buffer;
    if (config->inplace == 0) {
        vk.isInputFormatted = 1;
        vk.inputBuffer = &self->input_buffer;
        // Out-of-place r2c has a real side and a complex side, and the
        // inverse needs the real one as its destination. Without this, the
        // inverse result goes back through inputBufferStride[0] = size[0]
        // reals while the spectrum it reads lives in the size[0]/2 + 1
        // complex layout, so only the forward direction works. The flag
        // moves the last inverse write to inputBuffer, and it is read on
        // the inverse path only, so a forward-only plan is unaffected.
        // vkfft_execute pairs it with the matching slot assignment.
        vk.inverseReturnToInputBuffer = static_cast<pfUINT>(config->r2c != 0);
    }
    // bufferSize is deliberately left null. For every non-Vulkan backend
    // VkFFT derives it from size/numberBatches/precision, so there is no
    // buffer-size arithmetic here.

#if (VKFFT_BACKEND == 1)
    self->device = *static_cast<const CUdevice*>(device_handles[0]);
    vk.device = &self->device;
    vk.stream = &self->stream; // one stream, rewritten by every vkfft_execute call
    vk.num_streams = 1;
#elif (VKFFT_BACKEND == 3)
    self->platform = *static_cast<const cl_platform_id*>(device_handles[0]);
    self->device = *static_cast<const cl_device_id*>(device_handles[1]);
    self->context = *static_cast<const cl_context*>(device_handles[2]);
    self->queue = *static_cast<const cl_command_queue*>(device_handles[3]);
    vk.platform = &self->platform;
    vk.device = &self->device;
    vk.context = &self->context;
    vk.commandQueue = &self->queue; // a default that vkfft_execute overrides per call
#elif (VKFFT_BACKEND == 5)
    vk.device = static_cast<MTL::Device*>(device_handles[0]);
    vk.queue = static_cast<MTL::CommandQueue*>(device_handles[1]);
#endif

    const VkFFTResult res = initializeVkFFT(&self->app, vk);
    if (res != VKFFT_SUCCESS) {
        free(self); // initializeVkFFT already ran deleteVkFFT
        return static_cast<int>(res);
    }

    *out = self;
    return VKFFT_SUCCESS;
}

int vkfft_create(const vkfft_config* const config, void** const device_handles, vkfft_app** const out) {
    return create_app(config, device_handles, nullptr, 0, out);
}

int vkfft_create_loaded(const vkfft_config* const config, void** const device_handles, const char* const data,
                        const uint64_t size, vkfft_app** const out) {
    if (data == nullptr) {
        return VKFFT_WRAPPER_ERROR_INVALID_ARGUMENT;
    }
    return create_app(config, device_handles, data, size, out);
}

int vkfft_set_kernel(vkfft_app* const self, void* const kernel) {
    if ((self == nullptr) || (kernel == nullptr)) {
        return VKFFT_WRAPPER_ERROR_INVALID_ARGUMENT;
    }
    if (self->app.configuration.performConvolution == 0) {
        return VKFFT_WRAPPER_ERROR_INVALID_ARGUMENT; // nothing reads the slot on a plan that does no convolution
    }
    self->kernel_buffer = static_cast<buffer_handle>(kernel);
    return VKFFT_SUCCESS;
}

int vkfft_save(vkfft_app* const self, const char** const data, uint64_t* const size) {
    if ((self == nullptr) || (data == nullptr) || (size == nullptr)) {
        return VKFFT_WRAPPER_ERROR_INVALID_ARGUMENT;
    }
    *data = nullptr;
    *size = 0;

#if (VKFFT_BACKEND == 5)
    return VKFFT_WRAPPER_ERROR_UNSUPPORTED;
#else
    if (self->app.saveApplicationString == nullptr) {
        return VKFFT_ERROR_EMPTY_applicationString; // the plan was not created with save_to_string
    }
    // Borrowed, not copied. deleteVkFFT frees it, so it lives exactly as long
    // as the app does.
    *data = static_cast<const char*>(self->app.saveApplicationString);
    *size = static_cast<uint64_t>(self->app.applicationStringSize);
    return VKFFT_SUCCESS;
#endif
}

int vkfft_execute(vkfft_app* const self, void* const in, void* const out, const int direction, void* const stream) {
    if (self == nullptr) {
        return VKFFT_ERROR_EMPTY_app;
    }
    if (out == nullptr) {
        return VKFFT_ERROR_EMPTY_buffer;
    }

    const bool out_of_place = (self->app.configuration.isInputFormatted != 0);
    if (out_of_place && (in == nullptr)) {
        return VKFFT_ERROR_EMPTY_inputBuffer;
    }
    // VkFFT only checks that the slot exists, not that anything was put in it,
    // and an empty slot reaches the driver as a null buffer argument.
    if ((self->app.configuration.performConvolution != 0) && (self->kernel_buffer == buffer_handle())) {
        return VKFFT_ERROR_EMPTY_kernel;
    }

    // VkFFT normally treats inputBuffer as the source of both directions, so
    // `in` goes there. Under inverseReturnToInputBuffer the inverse instead
    // reads buffer and writes inputBuffer, so the two slots swap for that
    // direction to keep `in` the input of the call. VkFFTAppend reads
    // direction == 1 as inverse and everything else as forward.
    const bool returns_to_input = (self->app.configuration.inverseReturnToInputBuffer != 0);
    const bool swap_slots = out_of_place && returns_to_input && (direction == 1);

    self->buffer = static_cast<buffer_handle>(swap_slots ? in : out);
    if (out_of_place) {
        self->input_buffer = static_cast<buffer_handle>(swap_slots ? out : in);
    }

    VkFFTLaunchParams params = {};
    params.buffer = &self->buffer;
    if (out_of_place) {
        params.inputBuffer = &self->input_buffer;
    }

#if (VKFFT_BACKEND == 1)
    self->stream = static_cast<stream_handle>(stream); // no launch-params equivalent on CUDA
#elif (VKFFT_BACKEND == 3)
    if (stream == nullptr) {
        return VKFFT_ERROR_INVALID_QUEUE;
    }
    self->queue = static_cast<stream_handle>(stream);
    params.commandQueue = &self->queue;
#elif (VKFFT_BACKEND == 5)
    if (stream == nullptr) {
        return VKFFT_ERROR_INVALID_QUEUE;
    }
    MTL::CommandBuffer* const command_buffer = static_cast<stream_handle>(stream);
    // computeCommandEncoder hands back an autoreleased object, so the wrapper
    // does not own it and must not release it. A pool of our own keeps the
    // lifetime self-contained: the encoder is reclaimed here whether or not
    // the calling thread has a pool, and nothing is released twice when an
    // outer pool drains later.
    NS::AutoreleasePool* const pool = NS::AutoreleasePool::alloc()->init();
    MTL::ComputeCommandEncoder* const encoder = command_buffer->computeCommandEncoder();
    if (encoder == nullptr) {
        pool->release();
        return VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_LIST;
    }
    params.commandBuffer = command_buffer;
    params.commandEncoder = encoder;
#endif

    const VkFFTResult res = VkFFTAppend(&self->app, direction, &params);

#if (VKFFT_BACKEND == 5)
    // Ended even when the append failed, since a command buffer with an open
    // encoder cannot be committed. The caller commits.
    encoder->endEncoding();
    pool->release();
#endif

    return static_cast<int>(res);
}

void vkfft_destroy(vkfft_app* const self) {
    if (self == nullptr) {
        return;
    }
    deleteVkFFT(&self->app);
    free(self);
}

const char* vkfft_error_name(const int res) {
    if (res == VKFFT_WRAPPER_ERROR_INVALID_ARGUMENT) {
        return "VKFFT_WRAPPER_ERROR_INVALID_ARGUMENT";
    }
    if (res == VKFFT_WRAPPER_ERROR_UNSUPPORTED) {
        return "VKFFT_WRAPPER_ERROR_UNSUPPORTED";
    }
    return getVkFFTErrorString(static_cast<VkFFTResult>(res));
}

uint64_t vkfft_max_dims(void) {
    return static_cast<uint64_t>(VKFFT_MAX_FFT_DIMENSIONS);
}

uint64_t vkfft_backend(void) {
    return static_cast<uint64_t>(VKFFT_BACKEND);
}

uint64_t vkfft_config_size(void) {
    return static_cast<uint64_t>(sizeof(vkfft_config));
}

const char* vkfft_cuda_toolkit_root(void) {
#ifdef CUDA_TOOLKIT_ROOT_DIR
    // On a CUDA build vkFFT.h defines this itself when the build system did
    // not, so the empty string arrives here rather than an undefined macro.
    return CUDA_TOOLKIT_ROOT_DIR;
#else
    return ""; // no CUDA in this build, so there is no root to report
#endif
}
