// One translation unit for every backend. vkfft_wrapper.h is included first so
// that its VKFFT_MAX_FFT_DIMENSIONS default is in place before vkFFT.h picks
// its own.

#include "vkfft_wrapper.h"

#include <stdlib.h>

#include "vkFFT.h"

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

int vkfft_create(const vkfft_config* const config, void** const device_handles, vkfft_app** const out) {
    if ((config == nullptr) || (device_handles == nullptr) || (out == nullptr)) {
        return VKFFT_WRAPPER_ERROR_INVALID_ARGUMENT;
    }
    *out = nullptr;

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
    return getVkFFTErrorString(static_cast<VkFFTResult>(res));
}

uint64_t vkfft_max_dims(void) {
    return static_cast<uint64_t>(VKFFT_MAX_FFT_DIMENSIONS);
}

uint64_t vkfft_backend(void) {
    return static_cast<uint64_t>(VKFFT_BACKEND);
}
