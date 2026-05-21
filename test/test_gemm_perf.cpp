#include <iostream>
#include <vector>
#include <chrono>
#include <cassert>
#include <cstring>
#include "mat.h"
#include "allocator.h"

#ifdef NCNN_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

using namespace ncnn;

// naive CPU matmul using Mat API
static void cpu_matmul_mat(const Mat& A, const Mat& B, Mat& C, int N)
{
    // assume single channel
    for (int i = 0; i < N; ++i)
    {
        const float* arow = A.row(i);
        float* crow = C.row(i);
        for (int j = 0; j < N; ++j)
        {
            float s = 0.f;
            for (int k = 0; k < N; ++k)
                s += arow[k] * B.row(k)[j];
            crow[j] = s;
        }
    }
}

int main(int argc, char** argv)
{
    const int N = (argc > 1) ? atoi(argv[1]) : 1024;
    const size_t elem_count = (size_t)N * N;
    const size_t bytes = elem_count * sizeof(float);

    std::cout << "Matrix " << N << "x" << N << " (" << bytes << " bytes)" << std::endl;

    // host matrices via Mat API
    Mat A, B, C_cpu, C_gpu_host;
    A.create(N, N, 1);
    B.create(N, N, 1);
    C_cpu.create(N, N, 1);
    C_gpu_host.create(N, N, 1);

    // init
    for (size_t i = 0; i < elem_count; ++i)
    {
        ((float*)A.data)[i] = (float)(i & 255) / 255.f;
        ((float*)B.data)[i] = (float)((i * 7) & 255) / 255.f;
        ((float*)C_cpu.data)[i] = 0.f;
        ((float*)C_gpu_host.data)[i] = 0.f;
    }

    // CPU run
    auto t0 = std::chrono::high_resolution_clock::now();
    cpu_matmul_mat(A, B, C_cpu, N);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "CPU (Mat API) matmul time: " << cpu_ms << " ms" << std::endl;

#ifdef NCNN_CUDA
    auto alloc = get_current_gpu_allocator();
    if (!alloc)
    {
        std::cerr << "No GPU allocator available; ensure CUDA initialized." << std::endl;
    }
    else
    {
        // create device matrices from host Mats (this should copy H2D)
        CudaMat dA(A, alloc);
        CudaMat dB(B, alloc);
        CudaMat dC; dC.create(N, N, 4u, alloc);

        cublasHandle_t handle;
        cublasCreate(&handle);

        const float alpha = 1.0f;
        const float beta = 0.0f;

        // measure H2D + GEMM + D2H using CudaMat wrapper
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        // dA and dB were created from host Mats already (H2D done in constructor)
        // call cuBLAS: note row-major vs column-major handling
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha,
                    (const float*)dB.get_raw_data(), N,
                    (const float*)dA.get_raw_data(), N,
                    &beta,
                    (float*)dC.get_raw_data(), N);

        // copy device result back to host Mat buffer
        cudaMemcpy((void*)C_gpu_host.data, dC.get_raw_data(), bytes, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float gpu_ms = 0.f;
        cudaEventElapsedTime(&gpu_ms, start, stop);

        std::cout << "GPU (CudaMat + cuBLAS) time (H2D in construct + GEMM + D2H): " << gpu_ms << " ms" << std::endl;

        // verify
        double max_err = 0.0;
        for (size_t i = 0; i < elem_count; ++i)
        {
            double errd = std::abs((double)C_cpu[i] - (double)C_gpu_host[i]);
            if (errd > max_err) max_err = errd;
        }
        std::cout << "Max absolute error: " << max_err << std::endl;

        cublasDestroy(handle);
        cudaEventDestroy(start); cudaEventDestroy(stop);
    }
#else
    std::cout << "GPU test skipped (NCNN_CUDA not defined)" << std::endl;
#endif

    return 0;
}
