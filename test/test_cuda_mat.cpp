#include <cassert>
#include <iostream>
#include <memory>

#include "allocator.h"
#include "cuda_util.h"
#include "mat.h"

using namespace ncnn;

#if NCNN_CUDA
int main()
{
    const int w = 4000;
    const int h = 300;
    const int c = 200;

    Mat m(w, h, c);
    // Mat m(w);
    std::cout << "Mat elementsize =" << m.elemsize  << std::endl;
    assert(!m.empty());
    assert(m.total() == (size_t)w * h * c);

    m.fill(7.25f);

    std::shared_ptr<CudaAllocator> gpu_allocator = get_current_gpu_allocator();
    if (!gpu_allocator)
    {
        std::cout << "No CUDA allocator available; skipping CUDA mat test." << std::endl;
        return 0;
    }

    CudaMat cm(m, gpu_allocator);
    assert(!cm.empty());
    assert(cm.total() == m.total());
    auto host_data = cm.copy_gpu_data<float>();
    for (size_t i = 0; i < cm.total(); i++)
    {
        assert(host_data.get()[i] == 7.25f);
    }

    CudaMat cm_copy = cm;
    assert(!cm_copy.empty());

    CudaMat cm_copy1(cm);
    assert(!cm_copy1.empty());

    // CudaMat cm1(m, gpu_allocator);
    CudaMat cm_clone = cm.clone();
    std::cout << "CudaMat total = " << cm.total() << std::endl;
    std::cout << "CudaMat clone total = " << cm_clone.total() << std::endl;
    assert(!cm_clone.empty());
    cudaDeviceSynchronize();
    auto clone_data = cm_clone.copy_gpu_data<float>();
    for (size_t i = 0; i < cm_clone.total(); i++)
    {
        assert(clone_data.get()[i] == 7.25f);
    }

    CudaMat cm_like;
    cm_like.create_like(cm, gpu_allocator);
    cm_like.fill(1.0f);
    assert(cm_like.total() == cm.total());
    auto like_data = cm_like.copy_gpu_data<float>();
    for (size_t i = 0; i < cm_like.total(); i++)
    {
        assert(like_data.get()[i] == 1.0f);
    }

    std::cout << "test_cuda_mat passed" << std::endl;
    return 0;
    while(true) {
        // keep the program alive to allow inspection with cuda-memcheck or similar tools
    };
}
#else
    std::cout << "GPU test skipped (NCNN_CUDA not defined)" << std::endl;
#endif