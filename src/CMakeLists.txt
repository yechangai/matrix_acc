add_library(ncnn STATIC
    allocator.cpp
    gpu.cpp   
    mat.cpp
)
find_package(CUDAToolkit REQUIRED)
option(NCNN_CUDA "cuda compute support" ON)
option(NCNN_THREADS "build with threads" ON)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ncnn PRIVATE -march=native)
endif()

if (NCNN_CUDA)
    target_compile_definitions(ncnn PRIVATE NCNN_CUDA=1 NCNN_THREADS=1)
    target_link_libraries(ncnn PRIVATE CUDA::cudart)
endif()

if (NCNN_THREADS)
    target_compile_definitions(ncnn PRIVATE NCNN_THREADS=1)
endif()

target_include_directories(ncnn
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_features(ncnn
    PUBLIC
        cxx_std_17
)
