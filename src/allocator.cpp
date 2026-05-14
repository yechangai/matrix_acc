#include "allocator.h"

#include "gpu.h"
#include <chrono>

#include <algorithm>


#if NCNN_CUDA
#include "cuda_util.h"
#endif


namespace ncnn {

Allocator::~Allocator()
{
}

PoolAllocator::PoolAllocator()
{
    size_compare_ratio = 192; // 0.75f * 256
}

PoolAllocator::~PoolAllocator()
{
    clear();

    if (!payouts.empty())
    {
        printf("FATAL ERROR! pool allocator destroyed too early");
        std::list<std::pair<size_t, void*> >::iterator it = payouts.begin();
        for (; it != payouts.end(); ++it)
        {
            void* ptr = it->second;
            printf("%p still in use", ptr);
        }
    }
}

void PoolAllocator::clear()
{
    budgets_lock.lock();

    std::list<std::pair<size_t, void*> >::iterator it = budgets.begin();
    for (; it != budgets.end(); ++it)
    {
        void* ptr = it->second;
        ncnn::fastFree(ptr);
    }
    budgets.clear();

    budgets_lock.unlock();
}

void PoolAllocator::set_size_compare_ratio(float scr)
{
    if (scr < 0.f || scr > 1.f)
    {
        printf("invalid size compare ratio %f", scr);
        return;
    }

    size_compare_ratio = (unsigned int)(scr * 256);
}

void* PoolAllocator::fastMalloc(size_t size)
{
    budgets_lock.lock();

    // find free budget
    std::list<std::pair<size_t, void*> >::iterator it = budgets.begin();
    for (; it != budgets.end(); ++it)
    {
        size_t bs = it->first;

        // size_compare_ratio ~ 100%
        if (bs >= size && ((bs * size_compare_ratio) >> 8) <= size)
        {
            void* ptr = it->second;

            budgets.erase(it);

            budgets_lock.unlock();

            payouts_lock.lock();

            payouts.push_back(std::make_pair(bs, ptr));

            payouts_lock.unlock();

            return ptr;
        }
    }

    budgets_lock.unlock();

    // new
    void* ptr = ncnn::fastMalloc(size);

    payouts_lock.lock();

    payouts.push_back(std::make_pair(size, ptr));

    payouts_lock.unlock();

    return ptr;
}

void PoolAllocator::fastFree(void* ptr)
{
    payouts_lock.lock();

    // return to budgets
    std::list<std::pair<size_t, void*> >::iterator it = payouts.begin();
    for (; it != payouts.end(); ++it)
    {
        if (it->second == ptr)
        {
            size_t size = it->first;

            payouts.erase(it);

            payouts_lock.unlock();

            budgets_lock.lock();

            budgets.push_back(std::make_pair(size, ptr));

            budgets_lock.unlock();

            return;
        }
    }

    payouts_lock.unlock();

    printf("FATAL ERROR! pool allocator get wild %p", ptr);
    ncnn::fastFree(ptr);
}

UnlockedPoolAllocator::UnlockedPoolAllocator()
{
    size_compare_ratio = 192; // 0.75f * 256
}

UnlockedPoolAllocator::~UnlockedPoolAllocator()
{
    clear();

    if (!payouts.empty())
    {
        printf("FATAL ERROR! unlocked pool allocator destroyed too early");
        std::list<std::pair<size_t, void*> >::iterator it = payouts.begin();
        for (; it != payouts.end(); ++it)
        {
            void* ptr = it->second;
            printf("%p still in use", ptr);
        }
    }
}

void UnlockedPoolAllocator::clear()
{
    std::list<std::pair<size_t, void*> >::iterator it = budgets.begin();
    for (; it != budgets.end(); ++it)
    {
        void* ptr = it->second;
        ncnn::fastFree(ptr);
    }
    budgets.clear();
}

void UnlockedPoolAllocator::set_size_compare_ratio(float scr)
{
    if (scr < 0.f || scr > 1.f)
    {
        printf("invalid size compare ratio %f", scr);
        return;
    }

    size_compare_ratio = (unsigned int)(scr * 256);
}

void* UnlockedPoolAllocator::fastMalloc(size_t size)
{
    // find free budget
    std::list<std::pair<size_t, void*> >::iterator it = budgets.begin();
    for (; it != budgets.end(); ++it)
    {
        size_t bs = it->first;

        // size_compare_ratio ~ 100%
        if (bs >= size && ((bs * size_compare_ratio) >> 8) <= size)
        {
            void* ptr = it->second;

            budgets.erase(it);

            payouts.push_back(std::make_pair(bs, ptr));

            return ptr;
        }
    }

    // new
    void* ptr = ncnn::fastMalloc(size);

    payouts.push_back(std::make_pair(size, ptr));

    return ptr;
}

void UnlockedPoolAllocator::fastFree(void* ptr)
{
    // return to budgets
    std::list<std::pair<size_t, void*> >::iterator it = payouts.begin();
    for (; it != payouts.end(); ++it)
    {
        if (it->second == ptr)
        {
            size_t size = it->first;

            payouts.erase(it);

            budgets.push_back(std::make_pair(size, ptr));

            return;
        }
    }

    printf("FATAL ERROR! unlocked pool allocator get wild %p", ptr);
    ncnn::fastFree(ptr);
}

#if NCNN_CUDA
CudaAllocator::CudaAllocator(const CudaDevice* _cudev)
{
    cudev = _cudev;
    checkCudaErrors(cudaSetDevice(cudev->device_index));
}

void* CudaAllocator::fastMalloc(size_t size)
{
    void* buffer = nullptr;
    checkCudaErrors(cudaMalloc(&buffer, size));
    return buffer;
}

void CudaAllocator::fastFree(void* ptr)
{
    checkCudaErrors(cudaFree(ptr));
}

std::shared_ptr<ncnn::CudaAllocator> get_current_gpu_allocator()
{
    return std::shared_ptr<ncnn::CudaAllocator>{new ncnn::CudaAllocator(ncnn::get_current_gpu_device())};
}

CudaPoolAllocator::CudaPoolAllocator(const CudaDevice* _cudev)
    : CudaAllocator(_cudev)
{
    size_compare_ratio = 192; // 0.75f * 256
}

CudaPoolAllocator::~CudaPoolAllocator()
{
    clear();

    if (!payouts.empty())
    {
        printf("FATAL ERROR! cuda pool allocator destroyed too early");
        std::list<std::pair<size_t, void*> >::iterator it = payouts.begin();
        for (; it != payouts.end(); ++it)
        {
            void* ptr = it->second;
            printf("%p still in use", ptr);
        }
    }
}

void CudaPoolAllocator::clear()
{
    budgets_lock.lock();

    std::list<std::pair<size_t, void*> >::iterator it = budgets.begin();
    for (; it != budgets.end(); ++it)
    {
        void* ptr = it->second;
        CudaAllocator::fastFree(ptr);
    }
    budgets.clear();

    budgets_lock.unlock();
}

void CudaPoolAllocator::set_size_compare_ratio(float scr)
{
    if (scr < 0.f || scr > 1.f)
    {
        printf("invalid size compare ratio %f", scr);
        return;
    }

    size_compare_ratio = (unsigned int)(scr * 256);
}

void* CudaPoolAllocator::fastMalloc(size_t size)
{
    budgets_lock.lock();

    // find free budget
    std::list<std::pair<size_t, void*> >::iterator it = budgets.begin();
    for (; it != budgets.end(); ++it)
    {
        size_t bs = it->first;

        // size_compare_ratio ~ 100%
        if (bs >= size && ((bs * size_compare_ratio) >> 8) <= size)
        {
            void* ptr = it->second;

            budgets.erase(it);

            budgets_lock.unlock();

            payouts_lock.lock();

            payouts.push_back(std::make_pair(bs, ptr));

            payouts_lock.unlock();

            return ptr;
        }
    }

    budgets_lock.unlock();

    // new
    void* ptr = CudaAllocator::fastMalloc(size);

    payouts_lock.lock();

    payouts.push_back(std::make_pair(size, ptr));

    payouts_lock.unlock();

    return ptr;
}

void CudaPoolAllocator::fastFree(void* ptr)
{
    payouts_lock.lock();

    // return to budgets
    std::list<std::pair<size_t, void*> >::iterator it = payouts.begin();
    for (; it != payouts.end(); ++it)
    {
        if (it->second == ptr)
        {
            size_t size = it->first;

            payouts.erase(it);

            payouts_lock.unlock();

            budgets_lock.lock();

            budgets.push_back(std::make_pair(size, ptr));

            budgets_lock.unlock();

            return;
        }
    }

    payouts_lock.unlock();

    printf("FATAL ERROR! cuda pool allocator get wild %p", ptr);
    CudaAllocator::fastFree(ptr);
}




CudaUnlockedPoolAllocator::CudaUnlockedPoolAllocator(const CudaDevice* _cudev): CudaAllocator(_cudev)
{
    size_compare_ratio = 192; // 0.75f * 256
}

CudaUnlockedPoolAllocator::~CudaUnlockedPoolAllocator()
{
    clear();

    if (!payouts.empty())
    {
        printf("FATAL ERROR! unlocked pool allocator destroyed too early");
        std::list<std::pair<size_t, void*> >::iterator it = payouts.begin();
        for (; it != payouts.end(); ++it)
        {
            void* ptr = it->second;
            printf("%p still in use", ptr);
        }
    }
}

void CudaUnlockedPoolAllocator::clear()
{
    std::list<std::pair<size_t, void*> >::iterator it = budgets.begin();
    for (; it != budgets.end(); ++it)
    {
        void* ptr = it->second;
        CudaAllocator::fastFree(ptr);
    }
    budgets.clear();
}

void CudaUnlockedPoolAllocator::set_size_compare_ratio(float scr)
{
    if (scr < 0.f || scr > 1.f)
    {
        printf("invalid size compare ratio %f", scr);
        return;
    }

    size_compare_ratio = (unsigned int)(scr * 256);
}

void* CudaUnlockedPoolAllocator::fastMalloc(size_t size)
{
    // find free budget
    std::list<std::pair<size_t, void*> >::iterator it = budgets.begin();
    for (; it != budgets.end(); ++it)
    {
        size_t bs = it->first;

        // size_compare_ratio ~ 100%
        if (bs >= size && ((bs * size_compare_ratio) >> 8) <= size)
        {
            void* ptr = it->second;

            budgets.erase(it);

            payouts.push_back(std::make_pair(bs, ptr));

            return ptr;
        }
    }

    // new
    void* ptr = CudaAllocator::fastMalloc(size);

    payouts.push_back(std::make_pair(size, ptr));

    return ptr;
}

void CudaUnlockedPoolAllocator::fastFree(void* ptr)
{
    // return to budgets
    std::list<std::pair<size_t, void*> >::iterator it = payouts.begin();
    for (; it != payouts.end(); ++it)
    {
        if (it->second == ptr)
        {
            size_t size = it->first;

            payouts.erase(it);

            budgets.push_back(std::make_pair(size, ptr));

            return;
        }
    }

    printf("FATAL ERROR! unlocked pool allocator get wild %p", ptr);
    CudaAllocator::fastFree(ptr);
}


#endif


} // namespace ncnn
