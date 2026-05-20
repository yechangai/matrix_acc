#include <cassert>
#include <iostream>

#include "mat.h"

using namespace ncnn;

int main()
{
    const int w = 4;
    const int h = 3;
    const int c = 2;

    Mat m;
    assert(m.empty());

    m.create(w, h, c);
    assert(!m.empty());
    assert(m.total() == (size_t)w * h * c);
    assert(m.elemsize == 4u);
    assert(m.elempack == 1);

    m.fill(3.5f);
    for (size_t i = 0; i < m.total(); i++)
    {
        assert(m[i] == 3.5f);
    }

    Mat clone = m.clone();
    assert(clone.total() == m.total());
    for (size_t i = 0; i < clone.total(); i++)
    {
        assert(clone[i] == 3.5f);
    }

    Mat channel0 = m.channel(0);
    assert(channel0.total() == (size_t)w * h);
    for (size_t i = 0; i < channel0.total(); i++)
    {
        assert(channel0[i] == 3.5f);
    }

    float* row0 = m.row(0);
    assert(row0[0] == 3.5f);
    row0[0] = 5.0f;
    assert(m[0] == 5.0f);

    Mat copy = m;
    assert(copy.total() == m.total());
    assert(copy[0] == 5.0f);

    std::cout << "test_mat passed" << std::endl;
    return 0;
}
