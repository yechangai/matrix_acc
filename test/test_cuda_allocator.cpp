#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>

#if NCNN_CUDA
#include <cuda_runtime.h>
#include "allocator.h"
#include "gpu.h"

using namespace ncnn;

static bool verify_device_pattern(const std::vector<uint8_t>& data)
{
    for (size_t i = 0; i < data.size(); ++i)
    {
        uint8_t expected = static_cast<uint8_t>(i * 131u + 17u);
        if (data[i] != expected)
        {
            std::cerr << "mismatch at " << i << ": got " << static_cast<int>(data[i])
                      << ", expected " << static_cast<int>(expected) << std::endl;
            return false;
        }
    }
    return true;
}

int main()
{
    std::cout << "=== CUDA Allocator Test ===" << std::endl;

    try_initialize_cuda_gpu_instances();

    int gpu_count = get_cuda_gpu_count();
    std::cout << "cuda device count: " << gpu_count << std::endl;
    if (gpu_count <= 0)
    {
        std::cerr << "no CUDA device found" << std::endl;
        return 0;
    }

    int current_index = get_current_cuda_gpu_index();
    std::cout << "current cuda device index: " << current_index << std::endl;
    if (current_index < 0)
    {
        std::cerr << "failed to resolve current CUDA device" << std::endl;
        return 2;
    }

    const CudaGpuInfo info = get_cuda_gpu_info(current_index);
    std::cout << "device name: " << info.cuda_properties.name << std::endl;
    std::cout << "shared mem per block: " << info.cuda_properties.sharedMemPerBlock << std::endl;

    CudaDevice* device = get_current_gpu_device();
    if (!device)
    {
        std::cerr << "get_current_gpu_device() returned null" << std::endl;
        return 3;
    }
    std::cout << "selected device index: " << device->device_index << std::endl;

    auto alloc = get_current_gpu_allocator();
    if (!alloc)
    {
        std::cerr << "get_current_gpu_allocator() returned null" << std::endl;
        return 4;
    }

    const size_t bytes = 1 << 20;
    void* device_ptr = alloc->fastMalloc(bytes);
    if (!device_ptr)
    {
        std::cerr << "device allocation failed" << std::endl;
        return 5;
    }

    std::vector<uint8_t> host(bytes);
    std::vector<uint8_t> roundtrip(bytes, 0);
    for (size_t i = 0; i < host.size(); ++i)
    {
        host[i] = static_cast<uint8_t>(i * 131u + 17u);
    }

    cudaError_t err = cudaMemcpy(device_ptr, host.data(), bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        std::cerr << "cudaMemcpy H2D failed: " << cudaGetErrorString(err) << std::endl;
        alloc->fastFree(device_ptr);
        return 6;
    }

    err = cudaMemcpy(roundtrip.data(), device_ptr, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        std::cerr << "cudaMemcpy D2H failed: " << cudaGetErrorString(err) << std::endl;
        alloc->fastFree(device_ptr);
        return 7;
    }

    if (!verify_device_pattern(roundtrip))
    {
        alloc->fastFree(device_ptr);
        return 8;
    }

    alloc->fastFree(device_ptr);
    std::cout << "allocator roundtrip passed" << std::endl;
    return 0;
}
#else
int main()
{
    std::cerr << "NCNN_CUDA is not enabled for this build." << std::endl;
    return 1;
}
#endif
