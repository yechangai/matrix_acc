#ifndef ALLOCATOR_H
#define ALLOCATOR_H

#if NCNN_THREADS
#if (defined _WIN32 && !(defined __MINGW32__))
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <process.h>
#else
#include <pthread.h>
#endif
#endif // NCNN_THREADS

#include <stdlib.h>
#include <list>
#include <vector>
#include <string>

#if (defined _WIN32 && !(defined __MINGW32__))
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <process.h>
#else
#include <pthread.h>
#endif

#if NCNN_CUDA
#include <memory>     //引入 C++ 标准库的智能指针（std::shared_ptr/ std::unique_ptr）
#endif // NCNN_CUDA   

namespace ncnn {

#if NCNN_THREADS
#if (defined _WIN32 && !(defined __MINGW32__))
class Mutex
{
public:
    Mutex() { InitializeSRWLock(&srwlock); }
    ~Mutex() {}
    void lock() { AcquireSRWLockExclusive(&srwlock); }
    void unlock() { ReleaseSRWLockExclusive(&srwlock); }
private:
    friend class ConditionVariable;
    // NOTE SRWLock is available from windows vista
    SRWLOCK srwlock;
};

class ConditionVariable
{
public:
    ConditionVariable() { InitializeConditionVariable(&condvar); }
    ~ConditionVariable() {}
    void wait(Mutex& mutex) { SleepConditionVariableSRW(&condvar, &mutex.srwlock, INFINITE, 0); }
    void broadcast() { WakeAllConditionVariable(&condvar); }
    void signal() { WakeConditionVariable(&condvar); }
private:
    CONDITION_VARIABLE condvar;
};

static unsigned __stdcall start_wrapper(void* args);
class Thread
{
public:
    Thread(void* (*start)(void*), void* args = 0) { _start = start; _args = args; handle = (HANDLE)_beginthreadex(0, 0, start_wrapper, this, 0, 0); }
    ~Thread() {}
    void join() { WaitForSingleObject(handle, INFINITE); CloseHandle(handle); }
private:
    friend unsigned __stdcall start_wrapper(void* args)
    {
        Thread* t = (Thread*)args;
        t->_start(t->_args);
        return 0;
    }
    HANDLE handle;
    void* (*_start)(void*);
    void* _args;
};

class ThreadLocalStorage
{
public:
    ThreadLocalStorage() { key = TlsAlloc(); }
    ~ThreadLocalStorage() { TlsFree(key); }
    void set(void* value) { TlsSetValue(key, (LPVOID)value); }
    void* get() { return (void*)TlsGetValue(key); }
private:
    DWORD key;
};
#else // (defined _WIN32 && !(defined __MINGW32__))
class Mutex
{
public:
    Mutex() { pthread_mutex_init(&mutex, 0); }
    ~Mutex() { pthread_mutex_destroy(&mutex); }
    void lock() { pthread_mutex_lock(&mutex); }
    void unlock() { pthread_mutex_unlock(&mutex); }
private:
    friend class ConditionVariable;
    pthread_mutex_t mutex;
};

class ConditionVariable
{
public:
    ConditionVariable() { pthread_cond_init(&cond, 0); }
    ~ConditionVariable() { pthread_cond_destroy(&cond); }
    void wait(Mutex& mutex) { pthread_cond_wait(&cond, &mutex.mutex); }
    void broadcast() { pthread_cond_broadcast(&cond); }
    void signal() { pthread_cond_signal(&cond); }
private:
    pthread_cond_t cond;
};

class Thread
{
public:
    Thread(void* (*start)(void*), void* args = 0) { pthread_create(&t, 0, start, args); }
    ~Thread() {}
    void join() { pthread_join(t, 0); }
private:
    pthread_t t;
};

class ThreadLocalStorage
{
public:
    ThreadLocalStorage() { pthread_key_create(&key, 0); }
    ~ThreadLocalStorage() { pthread_key_delete(key); }
    void set(void* value) { pthread_setspecific(key, value); }
    void* get() { return pthread_getspecific(key); }
private:
    pthread_key_t key;
};
#endif // (defined _WIN32 && !(defined __MINGW32__))
#else // NCNN_THREADS
class Mutex
{
public:
    Mutex() {}
    ~Mutex() {}
    void lock() {}
    void unlock() {}
};

class ConditionVariable
{
public:
    ConditionVariable() {}
    ~ConditionVariable() {}
    void wait(Mutex& /*mutex*/) {}
    void broadcast() {}
    void signal() {}
};

class Thread
{
public:
    Thread(void* (*/*start*/)(void*), void* /*args*/ = 0) {}
    ~Thread() {}
    void join() {}
};

class ThreadLocalStorage
{
public:
    ThreadLocalStorage() {}
    ~ThreadLocalStorage() {}
    void set(void* /*value*/) {}
    void* get() { return 0; }
};
#endif // NCNN_THREADS

#if __AVX__
// 所有分配缓冲区的对齐大小
#define MALLOC_ALIGN 256
#else
// 所有分配缓冲区的对齐大小
#define MALLOC_ALIGN 16
#endif

// 将指针对齐到指定字节数
// ptr 对齐后的指针
// n 对齐大小，必须是2的幂
template<typename _Tp>
static inline _Tp* alignPtr(_Tp* ptr, int n = (int)sizeof(_Tp))
{
    return (_Tp*)(((size_t)ptr + n - 1) & -n);
}

// 将缓冲区大小对齐到指定字节数
// 函数返回大于或等于 sz 且能被 n 整除的最小数
// sz 要对齐的缓冲区大小
// n 对齐大小，必须是2的幂
static inline size_t alignSize(size_t sz, int n)
{
    return (sz + n - 1) & -n;
}

static inline void* fastMalloc(size_t size)
{
#if _MSC_VER
    return _aligned_malloc(size, MALLOC_ALIGN);
#elif (defined(__unix__) || defined(__APPLE__)) && _POSIX_C_SOURCE >= 200112L || (__ANDROID__ && __ANDROID_API__ >= 17)
    void* ptr = 0;
    if (posix_memalign(&ptr, MALLOC_ALIGN, size))
        ptr = 0;
    return ptr;
#else
    unsigned char* udata = (unsigned char*)malloc(size + sizeof(void*) + MALLOC_ALIGN);
    if (!udata)
        return 0;
    unsigned char** adata = alignPtr((unsigned char**)udata + 1, MALLOC_ALIGN);
    adata[-1] = udata;
    return adata;
#endif
}

static inline void fastFree(void* ptr)
{
    if (ptr)
    {
#if _MSC_VER
        _aligned_free(ptr);
#elif (defined(__unix__) || defined(__APPLE__)) && _POSIX_C_SOURCE >= 200112L || (__ANDROID__ && __ANDROID_API__ >= 17)
        free(ptr);
#else
        unsigned char* udata = ((unsigned char**)ptr)[-1];
        free(udata);
#endif
    }
}


#if NCNN_THREADS
// exchange-add operation for atomic operations on reference counters

#if defined(__INTEL_COMPILER) && !(defined(WIN32) || defined(_WIN32))
// Intel Compiler (Linux)
#define NCNN_XADD(addr, delta) \
    (int)_InterlockedExchangeAdd(const_cast<void*>(reinterpret_cast<volatile void*>(addr)), delta)

#elif defined(__GNUC__)
// GCC
#if defined(__ATOMIC_ACQ_REL)
#define NCNN_XADD(addr, delta) \
    (int)__atomic_fetch_add((unsigned*)(addr), (unsigned)(delta), __ATOMIC_ACQ_REL)
#else
#define NCNN_XADD(addr, delta) \
    (int)__sync_fetch_and_add((unsigned*)(addr), (unsigned)(delta))
#endif

#elif defined(_MSC_VER) && !defined(RC_INVOKED)
// MSVC
#define NCNN_XADD(addr, delta) \
    (int)_InterlockedExchangeAdd((long volatile*)addr, delta)

#else
// Thread-unsafe fallback
static inline int NCNN_XADD(int* addr, int delta)
{
    int tmp = *addr;
    *addr += delta;
    return tmp;
}
#endif
#endif // NCNN_THREADS


class Allocator
{
public:
    virtual ~Allocator();
    virtual void* fastMalloc(size_t size) = 0;
    virtual void fastFree(void* ptr) = 0;
};

class PoolAllocator : public Allocator
{
public:
    PoolAllocator();
    ~PoolAllocator();

    // ratio range 0 ~ 1
    // default cr = 0.75
    void set_size_compare_ratio(float scr);

    // release all budgets immediately
    void clear();

    virtual void* fastMalloc(size_t size);
    virtual void fastFree(void* ptr);

private:
    Mutex budgets_lock;
    Mutex payouts_lock;
    unsigned int size_compare_ratio; // 0~256
    std::list<std::pair<size_t, void*> > budgets;
    std::list<std::pair<size_t, void*> > payouts;
};

class UnlockedPoolAllocator : public Allocator
{
public:
    UnlockedPoolAllocator();
    ~UnlockedPoolAllocator();

    // ratio range 0 ~ 1
    // default cr = 0.75
    void set_size_compare_ratio(float scr);

    // release all budgets immediately
    void clear();

    virtual void* fastMalloc(size_t size);
    virtual void fastFree(void* ptr);

private:
    unsigned int size_compare_ratio; // 0~256
    std::list<std::pair<size_t, void*> > budgets;
    std::list<std::pair<size_t, void*> > payouts;
};


#if NCNN_CUDA

class CudaDevice;

class CudaAllocator  : public Allocator
{
public:
    CudaAllocator(const CudaDevice* _cudev);

    virtual ~CudaAllocator() {

    }

    virtual void* fastMalloc(size_t size);
    virtual void fastFree(void* ptr);

public:
    const CudaDevice* cudev;
};

std::shared_ptr<ncnn::CudaAllocator> get_current_gpu_allocator();

class CudaPoolAllocator : public CudaAllocator
{
public:
    CudaPoolAllocator(const CudaDevice* _cudev);
    ~CudaPoolAllocator();

    // ratio range 0 ~ 1
    // default cr = 0.75
    void set_size_compare_ratio(float scr);

    // release all budgets immediately
    void clear();

    virtual void* fastMalloc(size_t size);
    virtual void fastFree(void* ptr);

private:
    Mutex budgets_lock;
    Mutex payouts_lock;
    unsigned int size_compare_ratio; // 0~256
    std::list<std::pair<size_t, void*> > budgets;
    std::list<std::pair<size_t, void*> > payouts;
};

class CudaUnlockedPoolAllocator : public CudaAllocator
{
public:
    CudaUnlockedPoolAllocator(const CudaDevice* _cudev);
    ~CudaUnlockedPoolAllocator();

    // ratio range 0 ~ 1
    // default cr = 0.75
    void set_size_compare_ratio(float scr);

    // release all budgets immediately
    void clear();

    virtual void* fastMalloc(size_t size);
    virtual void fastFree(void* ptr);

private:
    unsigned int size_compare_ratio; // 0~256
    std::list<std::pair<size_t, void*> > budgets;
    std::list<std::pair<size_t, void*> > payouts;
};

#endif

}

#endif // NCNN_ALLOCATOR_H