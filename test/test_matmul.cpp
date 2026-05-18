#include <iostream>
#include <vector>
#include <chrono>
#include <cassert>
#include <cstring>
#include "allocator.h"

#ifdef NCNN_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

using namespace ncnn;

// naive CPU matmul
static void cpu_matmul(const float* A, const float* B, float* C, int N)
{
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            float s = 0.f;
            for (int k = 0; k < N; ++k)
                s += A[i * N + k] * B[k * N + j];
            C[i * N + j] = s;
        }
    }
}

int main()
{
    const int N = 256; // matrix dimension
    const size_t elem_count = (size_t)N * N;
    const size_t bytes = elem_count * sizeof(float);

    std::cout << "Matrix " << N << "x" << N << " (" << bytes << " bytes)" << std::endl;

    // host buffers via allocator and aligned
    float* A = (float*)fastMalloc(bytes + 64);
    float* B = (float*)fastMalloc(bytes + 64);
    float* C_cpu = (float*)fastMalloc(bytes + 64);
    float* C_gpu = (float*)fastMalloc(bytes + 64);

    if (!A || !B || !C_cpu || !C_gpu)
    {
        std::cerr << "host allocation failed" << std::endl;
        return 1;
    }

    A = alignPtr(A, 64);
    B = alignPtr(B, 64);
    C_cpu = alignPtr(C_cpu, 64);
    C_gpu = alignPtr(C_gpu, 64);

    // init
    for (size_t i = 0; i < elem_count; ++i)
    {
        A[i] = (float)(i & 255) / 255.f;
        B[i] = (float)((i * 7) & 255) / 255.f;
        C_cpu[i] = 0.f;
        C_gpu[i] = 0.f;
    }

    // CPU run
    auto t0 = std::chrono::high_resolution_clock::now();
    cpu_matmul(A, B, C_cpu, N);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "CPU matmul time: " << cpu_ms << " ms" << std::endl;

#ifdef NCNN_CUDA
    // GPU run using cuBLAS
    auto alloc = get_current_gpu_allocator();
    if (!alloc)
    {
        std::cerr << "No GPU allocator available; ensure CUDA initialized." << std::endl;
    }
    else
    {
        void* dA = alloc->fastMalloc(bytes);
        void* dB = alloc->fastMalloc(bytes);
        void* dC = alloc->fastMalloc(bytes);

        if (!dA || !dB || !dC)
        {
            std::cerr << "device allocation failed" << std::endl;
            alloc->fastFree(dA); alloc->fastFree(dB); alloc->fastFree(dC);
            return 2;
        }

        cudaMemcpy(dA, A, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(dB, B, bytes, cudaMemcpyHostToDevice);

        cublasHandle_t handle;
        cublasCreate(&handle);

        const float alpha = 1.0f;
        const float beta = 0.0f;

        // cublas uses column-major storage. Our data is row-major. To compute C = A*B (row-major),
        // compute C_col = B_col * A_col, so pass B as first operand and A as second.

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        cublasStatus_t stat = cublasSgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, N, N,
            &alpha,
            (const float*)dB, N,
            (const float*)dA, N,
            &beta,
            (float*)dC, N);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float gpu_ms = 0.f;
        cudaEventElapsedTime(&gpu_ms, start, stop);

        cudaMemcpy(C_gpu, dC, bytes, cudaMemcpyDeviceToHost);

        if (stat != CUBLAS_STATUS_SUCCESS)
        {
            std::cerr << "cublasSgemm failed: " << stat << std::endl;
        }

        std::cout << "GPU cuBLAS time (kernel only): " << gpu_ms << " ms" << std::endl;

        // measure full H2D + GEMM + D2H
        auto t0g = std::chrono::high_resolution_clock::now();
        void* dA2 = alloc->fastMalloc(bytes);
        void* dB2 = alloc->fastMalloc(bytes);
        void* dC2 = alloc->fastMalloc(bytes);
        cudaMemcpy(dA2, A, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(dB2, B, bytes, cudaMemcpyHostToDevice);
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, (const float*)dB2, N, (const float*)dA2, N, &beta, (float*)dC2, N);
        cudaMemcpy(C_gpu, dC2, bytes, cudaMemcpyDeviceToHost);
        auto t1g = std::chrono::high_resolution_clock::now();
        double total_gpu_ms = std::chrono::duration<double, std::milli>(t1g - t0g).count();

        std::cout << "GPU total (H2D+GEMM+D2H): " << total_gpu_ms << " ms" << std::endl;

        double max_err = 0.0;
        for (size_t i = 0; i < elem_count; ++i)
        {
            double errd = std::abs((double)C_cpu[i] - (double)C_gpu[i]);
            if (errd > max_err) max_err = errd;
        }
        std::cout << "Max absolute error: " << max_err << std::endl;

        alloc->fastFree(dA); alloc->fastFree(dB); alloc->fastFree(dC);
        alloc->fastFree(dA2); alloc->fastFree(dB2); alloc->fastFree(dC2);

        cublasDestroy(handle);
        cudaEventDestroy(start); cudaEventDestroy(stop);
    }
#else
    std::cout << "GPU test skipped (NCNN_CUDA not defined)" << std::endl;
#endif

    fastFree(A);
    fastFree(B);
    fastFree(C_cpu);
    fastFree(C_gpu);

    return 0;
}
