cmake_minimum_required(VERSION 3.20)

project(libvkfft
    VERSION 0.1.0
    LANGUAGES C CXX
    DESCRIPTION "Backend-generic C wrapper for VkFFT"
)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(VKFFT_BACKEND 3 CACHE STRING "VkFFT backend: 1 - CUDA, 3 - OpenCL, 5 - Metal")
set(VKFFT_MAX_FFT_DIMENSIONS 12 CACHE STRING "Maximum FFT dimensionality. Set to 12 for every build")
option(VKFFT_WRAPPER_BUILD_TESTS "Build the quick_test C executable" ON)
option(VKFFT_OPENCL_FORCE_ICD_LOADER "Link an ICD loader found by find_package(OpenCL) instead of the Apple framework" OFF)

set(vkfft_source_dir ${CMAKE_CURRENT_SOURCE_DIR}/lib/VkFFT)
if(NOT EXISTS ${vkfft_source_dir}/vkFFT/vkFFT.h)
    message(FATAL_ERROR "VkFFT sources not found in lib/VkFFT (pinned submodule not checked out?)")
endif()

add_library(vkfft SHARED src/vkfft_wrapper.cpp)
set_target_properties(vkfft PROPERTIES
    PUBLIC_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/include/vkfft_wrapper.h
    POSITION_INDEPENDENT_CODE ON
)
target_include_directories(vkfft
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>
)
# SYSTEM: VkFFT is header-only and warns a lot (sprintf, deprecated OpenCL
# 1.2 entry points). This is a problem for upstream and pollutes our
# diagnostics.
target_include_directories(vkfft SYSTEM PRIVATE ${vkfft_source_dir}/vkFFT)
target_compile_definitions(vkfft PRIVATE
    VKFFT_BACKEND=${VKFFT_BACKEND}
    VKFFT_MAX_FFT_DIMENSIONS=${VKFFT_MAX_FFT_DIMENSIONS}
)

if(VKFFT_BACKEND EQUAL 1)
    # No device code here. VkFFT emits CUDA source and compiles it with nvrtc
    # at runtime, so this stays plain C++ with no -gencode / architecture
    # list. It still needs cudart since VkFFT calls cudaMalloc/cudaFree for
    # its own temp buffer.
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        target_include_directories(vkfft SYSTEM PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
        target_link_libraries(vkfft PRIVATE CUDA::cuda_driver CUDA::nvrtc CUDA::cudart)
        set(vkfft_cuda_root ${CUDAToolkit_LIBRARY_ROOT})
    else()
        # FindCUDAToolkit insists on nvcc, which Yggdrasil builds do not have.
        # Locate the three pieces directly instead.
        find_path(VKFFT_CUDA_INCLUDE_DIR cuda.h REQUIRED)
        find_library(VKFFT_CUDA_DRIVER_LIBRARY cuda REQUIRED)
        find_library(VKFFT_CUDA_NVRTC_LIBRARY nvrtc REQUIRED)
        find_library(VKFFT_CUDA_RUNTIME_LIBRARY cudart REQUIRED)
        target_include_directories(vkfft SYSTEM PRIVATE ${VKFFT_CUDA_INCLUDE_DIR})
        target_link_libraries(vkfft PRIVATE
            ${VKFFT_CUDA_DRIVER_LIBRARY}
            ${VKFFT_CUDA_NVRTC_LIBRARY}
            ${VKFFT_CUDA_RUNTIME_LIBRARY}
        )
        set(vkfft_cuda_root "")
    endif()
    # VkFFT bakes this path into the fp16 kernels it generates, so it has to
    # be valid on the machine that runs the library, not the one that builds
    # it.
    # TODO: the JLL will need a runtime override here before fp16 works on CUDA.
    set(VKFFT_CUDA_TOOLKIT_ROOT_DIR "${vkfft_cuda_root}" CACHE STRING "Path baked into generated fp16 CUDA kernels")
    target_compile_definitions(vkfft PRIVATE CUDA_TOOLKIT_ROOT_DIR="${VKFFT_CUDA_TOOLKIT_ROOT_DIR}")
elseif(VKFFT_BACKEND EQUAL 3)
    if(APPLE AND NOT VKFFT_OPENCL_FORCE_ICD_LOADER)
        target_link_libraries(vkfft PRIVATE "-framework OpenCL")
        target_compile_definitions(vkfft PRIVATE CL_SILENCE_DEPRECATION)
    else()
        find_package(OpenCL REQUIRED)
        target_link_libraries(vkfft PRIVATE OpenCL::OpenCL)
    endif()
elseif(VKFFT_BACKEND EQUAL 5)
    if(NOT APPLE)
        message(FATAL_ERROR "VKFFT_BACKEND=5 (Metal) is macOS only")
    endif()
    # TODO: metal-cpp currently comes from the vendored VkFFT checkout. If we
    # ever want a newer metal-cpp than upstream ships, vendor it separately
    # and point this at it.
    if(NOT EXISTS ${vkfft_source_dir}/metal-cpp/Metal/Metal.hpp)
        message(FATAL_ERROR "metal-cpp not found in lib/VkFFT/metal-cpp")
    endif()
    target_include_directories(vkfft PRIVATE ${vkfft_source_dir}/metal-cpp)
    target_link_libraries(vkfft PRIVATE "-framework Foundation" "-framework QuartzCore" "-framework Metal")
else()
    message(FATAL_ERROR "Unsupported VKFFT_BACKEND '${VKFFT_BACKEND}': use 1 (CUDA), 3 (OpenCL) or 5 (Metal)")
endif()

if(VKFFT_WRAPPER_BUILD_TESTS AND VKFFT_BACKEND EQUAL 3)
    enable_testing()
    add_executable(vkfft_quick_test test/quick_test.c)
    target_link_libraries(vkfft_quick_test PRIVATE vkfft)
    # The test mirrors vkfft_config, so it must see the same array length as
    # the library.
    target_compile_definitions(vkfft_quick_test PRIVATE VKFFT_MAX_FFT_DIMENSIONS=${VKFFT_MAX_FFT_DIMENSIONS})
    if(APPLE AND NOT VKFFT_OPENCL_FORCE_ICD_LOADER)
        target_link_libraries(vkfft_quick_test PRIVATE "-framework OpenCL")
        target_compile_definitions(vkfft_quick_test PRIVATE CL_SILENCE_DEPRECATION)
    else()
        find_package(OpenCL REQUIRED)
        target_link_libraries(vkfft_quick_test PRIVATE OpenCL::OpenCL)
    endif()
    add_test(NAME quick_test COMMAND vkfft_quick_test)
endif()

include(GNUInstallDirs)
install(TARGETS vkfft
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
