#include <cassert>
#include <iostream>

#include "allocator.h"
#include "cuda_util.h"
#include "mat.h"

using namespace ncnn;

int main()
{
    const int w = 4;
    const int h = 3;
    const int c = 2;

    Mat m(w, h, c);
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

    CudaMat cm_clone = cm.clone();
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
}
