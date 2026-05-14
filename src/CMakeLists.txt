option(BUILD_CUDA "Enable CUDA support for ncnn" OFF)

set(ncnn_SRCS
    allocator.cpp
)

find_package(CUDAToolkit QUIET)
if (BUILD_CUDA AND CUDAToolkit_FOUND)
    list(APPEND ncnn_SRCS gpu.cpp)
endif()

add_library(ncnn STATIC ${ncnn_SRCS})

if (BUILD_CUDA AND CUDAToolkit_FOUND)
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
