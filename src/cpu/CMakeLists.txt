add_executable(cpu_benchmark
    main.cpp
)

target_include_directories(cpu_benchmark PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/cpu
)

if(BLAS_FOUND)
    target_compile_definitions(cpu_benchmark PRIVATE BLAS_FOUND)
    target_include_directories(cpu_benchmark PRIVATE ${BLAS_INCLUDE_DIRS})
    target_link_libraries(cpu_benchmark PRIVATE ${BLAS_LIBRARIES})
    if(APPLE)
        # Accelerate's cblas.h lives in vecLib.framework/Headers
        find_path(ACCELERATE_CBLAS_DIR cblas.h
            PATHS
                /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/Headers
                /System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/Headers
            NO_DEFAULT_PATH)
        if(ACCELERATE_CBLAS_DIR)
            target_include_directories(cpu_benchmark PRIVATE ${ACCELERATE_CBLAS_DIR})
        endif()
    endif()
endif()

if(EIGEN3_FOUND)
    target_compile_definitions(cpu_benchmark PRIVATE EIGEN3_FOUND)
    target_include_directories(cpu_benchmark PRIVATE ${EIGEN3_INCLUDE_DIR})
endif()
