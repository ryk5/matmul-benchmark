cmake_minimum_required(VERSION 3.16)
project(matmul_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Optimization flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    add_compile_options(-O3 -march=native -ffast-math)
    # Enable OpenMP if available
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        add_compile_options(${OpenMP_CXX_FLAGS})
        link_libraries(${OpenMP_CXX_LIBRARIES})
    endif()
endif()

if(MSVC)
    add_compile_options(/O2 /fp:fast /arch:AVX2)
endif()

# Dependencies
find_package(BLAS)
find_package(Eigen3 3.3 QUIET)

# CUDA (optional)
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit)
    message(STATUS "CUDA found: ${CMAKE_CUDA_COMPILER}")
else()
    message(STATUS "CUDA not found; GPU benchmarks will be disabled")
endif()

add_subdirectory(src/cpu)

if(CMAKE_CUDA_COMPILER)
    add_subdirectory(src/gpu)
endif()
