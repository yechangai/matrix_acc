set(ncnn_SRCS
    allocator.cpp
    gpu.cpp
        )

add_library(ncnn STATIC ${ncnn_SRCS})

target_include_directories(ncnn
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_features(ncnn
    PUBLIC
        cxx_std_17
)

# Test executable for allocator
add_executable(test_allocator test_allocator.cpp allocator.cpp)

target_include_directories(test_allocator
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_features(test_allocator
    PRIVATE
        cxx_std_17
)

enable_testing()
add_test(NAME allocator_test COMMAND test_allocator)
