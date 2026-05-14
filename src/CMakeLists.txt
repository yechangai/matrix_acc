add_library(ncnn STATIC
    allocator.cpp
)

target_include_directories(ncnn
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_features(ncnn
    PUBLIC
        cxx_std_17
)
