#include <iostream>
#include <vector>
#include <cassert>
#include <cstring>

#ifdef NCNN_CUDA
#include <cuda_runtime.h>
#include "allocator.h"

using namespace ncnn;

int main()
{
    std::cout << "=== Starting CUDA Allocator Test ===" << std::endl;

    // obtain current gpu allocator
    auto alloc = get_current_gpu_allocator();
    if (!alloc)
    {
        std::cerr << "get_current_gpu_allocator() returned null. Make sure CUDA is initialized." << std::endl;
        return 2;
    }

    const size_t sz = 4096;
    void* devptr = alloc->fastMalloc(sz);
    if (!devptr)
    {
        std::cerr << "fastMalloc returned null" << std::endl;
        return 3;
    }

    // prepare host buffer
    std::vector<unsigned char> host(sz);
    for (size_t i = 0; i < sz; ++i) host[i] = (unsigned char)(i & 0xFF);

    // copy host -> device
    cudaError_t err = cudaMemcpy(devptr, host.data(), sz, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        std::cerr << "cudaMemcpy H2D failed: " << cudaGetErrorString(err) << std::endl;
        alloc->fastFree(devptr);
        return 4;
    }

    // clear host and copy back
    std::fill(host.begin(), host.end(), 0);
    err = cudaMemcpy(host.data(), devptr, sz, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        std::cerr << "cudaMemcpy D2H failed: " << cudaGetErrorString(err) << std::endl;
        alloc->fastFree(devptr);
        return 5;
    }

    // verify
    for (size_t i = 0; i < sz; ++i)
    {
        unsigned char expected = (unsigned char)(i & 0xFF);
        if (host[i] != expected)
        {
            std::cerr << "Data mismatch at " << i << ": got " << (int)host[i] << " expected " << (int)expected << std::endl;
            alloc->fastFree(devptr);
            return 6;
        }
    }

    std::cout << "CUDA allocator H2D/D2H data verification passed." << std::endl;

    alloc->fastFree(devptr);
    std::cout << "Memory freed." << std::endl;

    std::cout << "=== CUDA Allocator Test Passed ===" << std::endl;
    return 0;
}
#else
int main()
{
    std::cerr << "NCNN_CUDA not defined; build with -DBUILD_CUDA=ON and CUDA available to run this test." << std::endl;
    return 1;
}
#endif
