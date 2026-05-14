extern "C" __global__ void matmul_kernel(const float* A, const float* B, float* C, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < N; ++k)
    {
        sum += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

extern "C" __host__ void gpu_matmul(const float* dA, const float* dB, float* dC, int N)
{
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    matmul_kernel<<<grid, block>>>(dA, dB, dC, N);
    cudaDeviceSynchronize();
}
