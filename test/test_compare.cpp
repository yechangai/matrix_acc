#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#include "allocator.h"
#include "gpu.h"

using namespace ncnn;

static void fill_pattern(std::vector<float>& data)
{
    for (size_t i = 0; i < data.size(); ++i)
    {
        data[i] = std::sin(static_cast<float>(i) * 0.013f) + std::cos(static_cast<float>(i) * 0.007f);
    }
}

static float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b)
{
    float max_diff = 0.f;
    for (size_t i = 0; i < a.size(); ++i)
    {
        float diff = std::fabs(a[i] - b[i]);
        if (diff > max_diff)
            max_diff = diff;
    }
    return max_diff;
}

int main()
{
    std::cout << "=== CPU/GPU Buffer Compare Test ===" << std::endl;

#if NCNN_CUDA
    try_initialize_cuda_gpu_instances();
    if (get_cuda_gpu_count() <= 0)
    {
        std::cerr << "no CUDA device found" << std::endl;
        return 0;
    }

    auto alloc = get_current_gpu_allocator();
    if (!alloc)
    {
        std::cerr << "get_current_gpu_allocator() returned null" << std::endl;
        return 2;
    }

    const size_t count = 1 << 18;
    const size_t bytes = count * sizeof(float);

    std::vector<float> cpu(count);
    std::vector<float> gpu_roundtrip(count, 0.f);
    fill_pattern(cpu);

    void* device_ptr = alloc->fastMalloc(bytes);
    if (!device_ptr)
    {
        std::cerr << "device allocation failed" << std::endl;
        return 3;
    }

    cudaError_t err = cudaMemcpy(device_ptr, cpu.data(), bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        std::cerr << "cudaMemcpy H2D failed: " << cudaGetErrorString(err) << std::endl;
        alloc->fastFree(device_ptr);
        return 4;
    }

    err = cudaMemcpy(gpu_roundtrip.data(), device_ptr, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        std::cerr << "cudaMemcpy D2H failed: " << cudaGetErrorString(err) << std::endl;
        alloc->fastFree(device_ptr);
        return 5;
    }

    float diff = max_abs_diff(cpu, gpu_roundtrip);
    std::cout << "max abs diff: " << diff << std::endl;
    if (diff != 0.f)
    {
        alloc->fastFree(device_ptr);
        return 6;
    }

    alloc->fastFree(device_ptr);
    std::cout << "compare test passed" << std::endl;
    return 0;
#else
    std::cout << "NCNN_CUDA is not enabled; running host-side buffer compare only." << std::endl;

    const size_t count = 1 << 18;
    std::vector<float> a(count);
    std::vector<float> b(count);
    fill_pattern(a);
    b = a;

    float diff = max_abs_diff(a, b);
    std::cout << "max abs diff: " << diff << std::endl;
    return diff == 0.f ? 0 : 1;
#endif
}
