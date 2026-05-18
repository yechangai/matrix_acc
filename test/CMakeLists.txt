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

# Separate matmul benchmark test
if (BUILD_CUDA AND CUDAToolkit_FOUND)
    add_executable(test_matmul
        test_matmul.cpp
    )

    target_include_directories(test_matmul
        PRIVATE
            ${PROJECT_SOURCE_DIR}/src
    )

    target_link_libraries(test_matmul
        PRIVATE
            ncnn
            CUDA::cudart
            CUDA::cublas
    )

    target_compile_definitions(test_matmul PRIVATE NCNN_CUDA=1)

    add_test(NAME matmul_test COMMAND test_matmul)
else()
    add_executable(test_matmul_cpu
        test_matmul.cpp
    )

    target_include_directories(test_matmul_cpu
        PRIVATE
            ${PROJECT_SOURCE_DIR}/src
    )

    target_link_libraries(test_matmul_cpu
        PRIVATE
            ncnn
    )

    add_test(NAME matmul_cpu_test COMMAND test_matmul_cpu)
endif()

    # CPU / GPU compare test
    if (BUILD_CUDA AND CUDAToolkit_FOUND)
        add_executable(test_compare
            test_compare.cpp
        )

        target_include_directories(test_compare
            PRIVATE
                ${PROJECT_SOURCE_DIR}/src
        )

        target_link_libraries(test_compare
            PRIVATE
                ncnn
                CUDA::cudart
                CUDA::cublas
        )

        target_compile_definitions(test_compare PRIVATE NCNN_CUDA=1)

        add_test(NAME compare_test COMMAND test_compare)
    else()
        add_executable(test_compare_cpu
            test_compare.cpp
        )

        target_include_directories(test_compare_cpu
            PRIVATE
                ${PROJECT_SOURCE_DIR}/src
        )

        target_link_libraries(test_compare_cpu
            PRIVATE
                ncnn
        )

        add_test(NAME compare_cpu_test COMMAND test_compare_cpu)
    endif()