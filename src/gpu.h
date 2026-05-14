#ifndef NCNN_GPU_H
#define NCNN_GPU_H

#if NCNN_CUDA

#include "cuda_util.h"

namespace ncnn {

struct CudaGpuInfo {
    int type{-1};
    cudaDeviceProp cuda_properties{};
};


class CudaDevice
{
public:
	CudaDevice(int device_index);
	~CudaDevice();

	int device_index{-1};
    CudaGpuInfo info{};


};


void try_initialize_cuda_gpu_instances();
int get_cuda_gpu_count();
int get_current_cuda_gpu_index();
const CudaGpuInfo get_cuda_gpu_info(int device_index);
CudaDevice* get_cuda_gpu_device(int cuda_device_index);
CudaDevice* get_current_gpu_device();


}
#endif // NCNN_CUDA

#endif // NCNN_GPU_H
