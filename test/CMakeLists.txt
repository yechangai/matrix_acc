cmake_minimum_required(VERSION 3.20)

# Test executable for allocator
add_executable(test_allocator
    test_allocator.cpp
)

target_include_directories(test_allocator
    PRIVATE
        ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(test_allocator
    PRIVATE
        ncnn
)

add_test(NAME allocator_test COMMAND test_allocator)

# CUDA allocator test (only when BUILD_CUDA is enabled and CUDAToolkit is found)
find_package(CUDAToolkit QUIET)
if (BUILD_CUDA AND CUDAToolkit_FOUND)
    add_executable(test_cuda_allocator
        test_cuda_allocator.cpp
    )

    target_include_directories(test_cuda_allocator
        PRIVATE
            ${PROJECT_SOURCE_DIR}/src
    )

    target_link_libraries(test_cuda_allocator
        PRIVATE
            ncnn
            CUDA::cudart
    )

    # ensure test compilation sees NCNN_CUDA
    target_compile_definitions(test_cuda_allocator PRIVATE NCNN_CUDA=1)

    add_test(NAME cuda_allocator_test COMMAND test_cuda_allocator)
endif()