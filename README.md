# libvkfft

A thin C wrapper for [VkFFT](https://github.com/DTolm/VkFFT), built once per
backend. This wrapper is essentially just a stable C ABI for
[VkFFT.jl](https://github.com/PaulVirally/VkFFT.jl). You probably don't want to
install this library directly, rather use VkFFT.jl and the `VkFFT_*_jll`
packages (built on top of this library) instead.

The backend in VkFFT is specified at compile-time and changes struct layouts, so
one shared library per backend is unavoidable. This wrapper is built to be
generic for all backends, and thus `include/vkfft_wrapper.h` and
`src/vkfft_wrapper.cpp` are compiled three times with different `VKFFT_BACKEND`
values. Every build exports the same eleven symbols with the same signatures:

- `vkfft_create`: create a plan
- `vkfft_create_loaded`: create a plan from binaries saved by `vkfft_save`
- `vkfft_execute`: execute a plan
- `vkfft_set_kernel`: point a convolution plan at its pre-transformed kernel
- `vkfft_save`: get the compiled kernel binaries of a plan built with `save_to_string`
- `vkfft_destroy`: destroy a plan
- `vkfft_error_name`: get a string for an error code
- `vkfft_max_dims`: get the maximum number of FFT dimensions (set to 12 by default)
- `vkfft_config_size`: get `sizeof(vkfft_config)`, so a mirror can check its ABI
- `vkfft_backend`: get the backend the library was built for (1=CUDA, 3=OpenCL, 5=Metal)
- `vkfft_cuda_toolkit_root`: get the `CUDA_TOOLKIT_ROOT_DIR` baked in at compile time, empty when there is none

## Vendored VkFFT

`lib/VkFFT` is VkFFT v1.3.4, the latest release tag as of August 2026 ([commit
`066a17c17068c0f11c9298d848c2976c71fad1c1`](https://github.com/DTolm/VkFFT/tree/066a17c17068c0f11c9298d848c2976c71fad1c1)).

All builds use `VKFFT_MAX_FFT_DIMENSIONS=12`. Callers should read
`vkfft_max_dims()` instead of assuming 12 in case we change this in the future
(or the library is built with a different value).

## Building

```sh
cmake -S . -B build -DVKFFT_BACKEND=3 -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

`VKFFT_BACKEND` is `1` for CUDA, `3` for OpenCL, `5` for Metal.
`VKFFT_MAX_FFT_DIMENSIONS` is a cache variable too, but changing it breaks ABI
compatibility with the Julia side.


| Backend | Requirements |
| --- | --- |
| CUDA (`1`) | CUDA headers, `libcuda`, `libnvrtc`, `libcudart`. No device code is compiled, so no architecture list is needed (VkFFT builds its kernels through nvrtc at runtime). If `find_package(CUDAToolkit)` fails because of a lack of `nvcc`, the build falls back to locating the three libraries directly. |
| OpenCL (`3`) | An OpenCL 1.2 implementation. On macOS, we use `-framework OpenCL` by default. For every other target, and on macOS when `VKFFT_OPENCL_FORCE_ICD_LOADER` is `ON`, we use `find_package(OpenCL)`. |
| Metal (`5`) | macOS, plus the `metal-cpp` headers in `lib/VkFFT/metal-cpp`. |

## Testing

`test/quick_test.c` runs fp32 transforms on the first OpenCL device it finds and
compares them against a naive double-precision DFT computed on the host: a 1D
c2c of length 64, then a table of r2c cases covering both directions in 1D and
2D, even and odd real axes, batches, an omitted axis, and a forward-only plan.
Both directions are checked in-place and out-of-place, with every out-of-place
destination poisoned first so a transform that writes nothing cannot pass.

`test/quick_test_metal.cpp` is the Metal counterpart, written against metal-cpp:
c2c in 1D, 2D and batched, sizes large enough to need several shared-memory
uploads per axis, a prime size that goes through Bluestein, out-of-place r2c
round trips with an even and an odd real axis, plan reuse across command buffers
and inside one command buffer, and a stress loop that watches the process
footprint across thousands of executes so that an encoder leaked or released
twice shows up.

One test is built per backend and registered with ctest:

```sh
cd build && ctest --output-on-failure
# or just: ./vkfft_quick_test, ./vkfft_quick_test_metal
```

## `device_handles`

`vkfft_create` takes `void** device_handles`, whose length and contents depend on the
backend the library was built for:

| Backend | `device_handles` |
| --- | --- |
| CUDA | `{ CUdevice* }` |
| OpenCL | `{ cl_platform_id*, cl_device_id*, cl_context*, cl_command_queue* }` |
| Metal | `{ MTLDevice*, MTLCommandQueue* }` |

For CUDA and OpenCL the entries are pointers to handles, which is what VkFFT
itself wants. For Metal they are the objects. The wrapper copies each value into
storage it owns, so the caller's variables may go out of scope right after the
call.

Metal adds two obligations to `vkfft_create`. It submits to the queue and waits
while it builds its tables, so it is a synchronizing call. And VkFFT's Metal plan
builder over-releases the strings it compiles kernels from, so it must not run
inside an autorelease pool that is drained afterwards: drain any pool the thread
holds first, or call from a thread that has none. `vkfft_execute` is unaffected.
Upstream is aware (the string over-releases are among the fixes in
[DTolm/VkFFT#227](https://github.com/DTolm/VkFFT/pull/227), unmerged as of
August 2026). If the vendored VkFFT is ever bumped past that fix, the obligation
inverts: creates stop crashing under a pool but leak their strings without one,
so this wrapper should then open a pool around `vkfft_create` as it already does
around `vkfft_execute`.

The OpenCL command queue passed here is only a default. `vkfft_execute` takes
the queue to submit on as its `stream` argument and overrides it on every call,
which is what lets one plan serve several queues (on CUDA this means several
per-task streams).

## Buffer handles in `vkfft_execute`

`in` and `out` are device buffer handles. The representation of these handles
differs per backend:

| Backend | `in` / `out` | `stream` |
| --- | --- | --- |
| CUDA | device pointer | `cudaStream_t` |
| OpenCL | `cl_mem` | `cl_command_queue` |
| Metal | `MTLBuffer` | `MTLCommandBuffer` |

Note that `cl_mem` is itself a pointer type, so it can be cast through `void*`
without any problems. VkFFT wants a `cl_mem*` in its launch parameters, so the
wrapper supplies the address of its own slot.

On Metal the wrapper opens a compute encoder on the command buffer you give it,
appends the FFT, and ends the encoding. It never commits so that submission and
synchronization are the responsibility of the caller. Several calls may share one
command buffer before a single commit, and the encoders run in the order they
were made. The encoder is the wrapper's, reclaimed by an autorelease pool the
call owns, so it is gone by the time the call returns. The command buffer is
yours and comes back from the queue autoreleased, so hold a pool rather than
release it by hand.

There is no offset in the Metal launch parameters, so `in` and `out` have to be
buffers whose first element is the first element of the transform, the same
restriction OpenCL has.

`direction` is the same as `VkFFTAppend`: `-1` forward, `1` inverse. The forward
transform uses the `exp(-2*pi*i*k*n/N)` convention, and the inverse is
unnormalized unless `config.normalize` is set.

For an in-place plan, VkFFT works out of a single buffer, so `vkfft_execute`
uses `out` and ignores `in`. A caller should thus pass the same handle twice.

An out-of-place r2c plan has a real side and a complex side, and which handle
holds which follows the direction. The forward transform reads reals from `in`
and writes the spectrum to `out`, and the inverse reads the spectrum from `in`
and writes reals to `out`. Per batch the real side holds `prod(size)` reals and
the complex side holds `(size[0]/2 + 1) * prod(size[1:])` complex values. Both
are dense in their own element type, so the real side carries none of the row
padding the in-place layout needs. The inverse uses the spectrum as scratch
space, so treat `in` as clobbered unless a single axis is transformed.
