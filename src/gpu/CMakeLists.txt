add_executable(gpu_benchmark
    main.cu
)

set_target_properties(gpu_benchmark PROPERTIES
    CUDA_ARCHITECTURES "75;80;86;89"
)

target_include_directories(gpu_benchmark PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/gpu
)

if(CUDAToolkit_FOUND)
    target_link_libraries(gpu_benchmark PRIVATE
        CUDA::cudart
        CUDA::cublas
    )
endif()
