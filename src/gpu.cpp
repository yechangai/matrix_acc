#include "gpu.h"

#include <atomic>
#include <allocator.h>
#include <mutex>

#if NCNN_CUDA

namespace ncnn {


	CudaDevice::CudaDevice(int _device_index)
	{
		device_index = _device_index;
        info = get_cuda_gpu_info(device_index);
	}

	CudaDevice::~CudaDevice()
	{

	}

static int g_cuda_gpu_count = 0;
static int g_current_cuda_gpu_index = -1;

#define NCNN_MAX_CUDA_GPU_COUNT 8

// default cuda device
static Mutex g_default_cudev_lock;
static std::atomic_bool cuda_gpu_instances_initialized{false};
static CudaGpuInfo g_cuda_gpu_infos[NCNN_MAX_CUDA_GPU_COUNT];
static CudaDevice* g_cuda_devices[NCNN_MAX_CUDA_GPU_COUNT] = {};

const CudaGpuInfo get_cuda_gpu_info(int device_index)
{
    try_initialize_cuda_gpu_instances();

    return g_cuda_gpu_infos[device_index];
}

CudaDevice* get_cuda_gpu_device(int cuda_device_index)
{
    try_initialize_cuda_gpu_instances();

    if (cuda_device_index < 0 || cuda_device_index >= g_cuda_gpu_count)
        return nullptr;

    MutexLockGuard lock(g_default_cudev_lock);

    if (!g_cuda_devices[cuda_device_index])
    {
        if (cuda_device_index > get_cuda_gpu_count() - 1)
        {
            return 0;
        }

        g_cuda_devices[cuda_device_index] = new CudaDevice(cuda_device_index);
    }

    return g_cuda_devices[cuda_device_index];
}

CudaDevice* get_current_gpu_device()
{
    return  get_cuda_gpu_device(get_current_cuda_gpu_index());
}

int get_current_cuda_gpu_index()
{
    if (g_current_cuda_gpu_index < 0) {
        try_initialize_cuda_gpu_instances();
    }

    return g_current_cuda_gpu_index;
}

static int find_default_cuda_device_index()
{
    // first try, discrete gpu
    for (int i = 0; i < g_cuda_gpu_count; i++)
    {
        if (g_cuda_gpu_infos[i].type == 0)
            return i;
    }

    // second try, integrated gpu
    for (int i = 0; i < g_cuda_gpu_count; i++)
    {
        if (g_cuda_gpu_infos[i].type == 1)
            return i;
    }

    // third try, any probed device
    if (g_cuda_gpu_count > 0)
        return 0;

    printf("no cuda device");
    return -1;
}


static int initialize_cuda_gpu_instances()
{
    MutexLockGuard lock(g_default_cudev_lock);

    // find proper device and queue
    int cuda_gpu_info_index = 0;
    for (int i = 0; i < get_cuda_gpu_count(); i++)
    {
        cudaDeviceProp prop;
        checkCudaErrors(cudaGetDeviceProperties(&prop, i));
        std::cout << "Cuda device Number:" << i << std::endl;
        std::cout << " Device name: " << prop.name << std::endl;
        // #if CUDA_VERSION >= 12000
        //     std::cout << " Device Memory Clock Rate (KHz): " << prop.memoryClockRate << std::endl;
        // #else
        //     // 旧版本 CUDA 使用替代方法
        //     int memoryClockRate;
        //     cudaDeviceGetAttribute(&memoryClockRate, cudaDevAttrMemoryClockRate, device);
        //     std::cout << " Device Memory Clock Rate (KHz): " << memoryClockRate << std::endl;
        // #endif
        std::cout << " Device Memory Bus Width (bits): " << prop.memoryBusWidth << std::endl;
        std::cout << " Device Peak Memory Bandwidth (GB/s): " << prop.memoryBusWidth << std::endl;
        std::cout << " Device Shared Memory Per Block (GB/s): " << prop.sharedMemPerBlock << std::endl;

        g_cuda_gpu_infos[cuda_gpu_info_index].cuda_properties = prop;
        g_cuda_gpu_infos[cuda_gpu_info_index].type = 0; //todo differentiate types?

        cuda_gpu_info_index++;
    }

    g_cuda_gpu_count = cuda_gpu_info_index;

    if (g_cuda_gpu_count > 0)
    {
        // the default gpu device
        g_current_cuda_gpu_index = find_default_cuda_device_index();
    }

    return 0;
}

void try_initialize_cuda_gpu_instances()
{
    if (!cuda_gpu_instances_initialized)
    {
        cuda_gpu_instances_initialized = true;
        initialize_cuda_gpu_instances();
    }
}


int get_cuda_gpu_count()
{
    int cuda_device_count = 0;
    checkCudaErrors(cudaGetDeviceCount(&cuda_device_count));
    return cuda_device_count;
}

}
#endif

