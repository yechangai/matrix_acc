add_library(ncnn STATIC
    allocator.cpp
    gpu.cpp   
)
find_package(CUDAToolkit REQUIRED)
if (BUILD_CUDA)
    target_compile_definitions(ncnn PRIVATE NCNN_CUDA=1)
    target_link_libraries(ncnn PRIVATE CUDA::cudart)
endif()

target_include_directories(ncnn
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_features(ncnn
    PUBLIC
        cxx_std_17
)
