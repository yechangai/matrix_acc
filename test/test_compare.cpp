#include <iostream>
#include <vector>
#include <chrono>
#include <cassert>
#include <cstring>
#include "allocator.h"

using namespace ncnn;

#ifdef NCNN_CUDA
#include <cuda_runtime.h>
#ifdef __cplusplus
extern "C" void gpu_matmul(const float* dA, const float* dB, float* dC, int N);
#else
void gpu_matmul(const float* dA, const float* dB, float* dC, int N);
#endif
#endif

#ifdef NCNN_CUDA
#include <cublas_v2.h>
#endif

// simple CPU matrix multiplication
void cpu_matmul(const float* A, const float* B, float* C, int N)
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
    const int N = 1024; // adjust as needed
    const size_t sz = (size_t)N * N * sizeof(float);

    std::cout << "Matrix size: " << N << "x" << N << std::endl;

    // allocate aligned host buffers using fastMalloc
    float* A = (float*)fastMalloc(sz + 64);
    float* B = (float*)fastMalloc(sz + 64);
    float* C_cpu = (float*)fastMalloc(sz + 64);
    float* C_gpu = (float*)fastMalloc(sz + 64);

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
    for (int i = 0; i < N * N; ++i)
    {
        A[i] = (float)(i % 256) / 255.f;
        B[i] = (float)((i * 7) % 256) / 255.f;
        C_cpu[i] = 0.f;
        C_gpu[i] = 0.f;
    }

    // CPU
    auto t0 = std::chrono::high_resolution_clock::now();
    cpu_matmul(A, B, C_cpu, N);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "CPU matmul time: " << cpu_ms << " ms" << std::endl;

#ifdef NCNN_CUDA
    // GPU path
    auto alloc = get_current_gpu_allocator();
    if (!alloc)
    {
        std::cerr << "no gpu allocator available" << std::endl;
    }
    else
    {
        void* dA = alloc->fastMalloc(sz);
        void* dB = alloc->fastMalloc(sz);
        void* dC = alloc->fastMalloc(sz);

        if (!dA || !dB || !dC)
        {
            std::cerr << "device allocation failed" << std::endl;
            alloc->fastFree(dA);
            alloc->fastFree(dB);
            alloc->fastFree(dC);
            return 2;
        }

        // copy H->D
        cudaMemcpy(dA, A, sz, cudaMemcpyHostToDevice);
        cudaMemcpy(dB, B, sz, cudaMemcpyHostToDevice);

        auto t2 = std::chrono::high_resolution_clock::now();

        // call GPU matmul via cuBLAS: C = A * B (row-major host -> use transposed ops)
        cublasHandle_t handle;
        cublasCreate(&handle);

        const float alpha = 1.0f;
        const float beta = 0.0f;
        // Use cublasSgemm with transpose flags to handle row-major data: C = A * B
        // cublasSgemm(handle, transB, transA, N, N, N, &alpha, dB, N, dA, N, &beta, dC, N);
        cublasStatus_t st = cublasSgemm(handle,
            CUBLAS_OP_T, CUBLAS_OP_T,
            N, N, N,
            &alpha,
            (const float*)dB, N,
            (const float*)dA, N,
            &beta,
            (float*)dC, N);

        if (st != CUBLAS_STATUS_SUCCESS)
        {
            std::cerr << "cublasSgemm failed: " << st << std::endl;
            cublasDestroy(handle);
            alloc->fastFree(dA);
            alloc->fastFree(dB);
            alloc->fastFree(dC);
            return 7;
        }

        cudaDeviceSynchronize();
        cublasDestroy(handle);

        // copy D->H
        cudaMemcpy(C_gpu, dC, sz, cudaMemcpyDeviceToHost);

        auto t3 = std::chrono::high_resolution_clock::now();
        double gpu_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

        std::cout << "GPU matmul time (kernel + H2D/D2H): " << gpu_ms << " ms" << std::endl;

        // verify
        double max_err = 0.0;
        for (int i = 0; i < N * N; ++i)
        {
            double err = std::abs(C_cpu[i] - C_gpu[i]);
            if (err > max_err) max_err = err;
        }
        std::cout << "Max absolute error: " << max_err << std::endl;

        alloc->fastFree(dA);
        alloc->fastFree(dB);
        alloc->fastFree(dC);
    }
#else
    std::cout << "GPU test skipped: build with -DBUILD_CUDA=ON" << std::endl;
#endif

    // free host
    fastFree(A);
    fastFree(B);
    fastFree(C_cpu);
    fastFree(C_gpu);

    return 0;
}
