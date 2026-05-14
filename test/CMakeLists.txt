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