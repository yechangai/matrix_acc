#include <iostream>
#include <cstring>
#include <cassert>
#include "allocator.h"

using namespace ncnn;

void test_alignPtr()
{
    std::cout << "Testing alignPtr..." << std::endl;
    
    // Test 1: Align pointer to 16 bytes
    unsigned char buffer[256];
    unsigned char* ptr = buffer + 5;  // Unaligned pointer
    
    unsigned char* aligned_ptr = alignPtr(ptr, 16);
    size_t addr = (size_t)aligned_ptr;
    assert(addr % 16 == 0);
    std::cout << "  Original ptr: " << (size_t)ptr << std::endl;
    std::cout << "  Aligned ptr (16): " << addr << std::endl;
    assert(aligned_ptr >= ptr);
    
    // Test 2: Align pointer to 256 bytes
    unsigned char* aligned_ptr_256 = alignPtr(ptr, 256);
    addr = (size_t)aligned_ptr_256;
    assert(addr % 256 == 0);
    std::cout << "  Aligned ptr (256): " << addr << std::endl;
    assert(aligned_ptr_256 >= ptr);
    
    // Test 3: Align already aligned pointer
    unsigned char* already_aligned = alignPtr(buffer, 16);
    assert((size_t)already_aligned % 16 == 0);
    
    std::cout << "  alignPtr tests passed!" << std::endl;
}

void test_alignSize()
{
    std::cout << "Testing alignSize..." << std::endl;
    
    // Test 1: Align size to 16 bytes
    size_t sz = 100;
    size_t aligned = alignSize(sz, 16);
    assert(aligned % 16 == 0);
    assert(aligned >= sz);
    std::cout << "  Original size: " << sz << ", aligned (16): " << aligned << std::endl;
    
    // Test 2: Align to 256 bytes
    sz = 1000;
    aligned = alignSize(sz, 256);
    assert(aligned % 256 == 0);
    std::cout << "  Original size: " << sz << ", aligned (256): " << aligned << std::endl;
    
    // Test 3: Already aligned size
    sz = 64;
    aligned = alignSize(sz, 16);
    assert(aligned == sz);
    
    std::cout << "  alignSize tests passed!" << std::endl;
}

void test_fastMalloc_fastFree()
{
    std::cout << "Testing fastMalloc and fastFree..." << std::endl;
    
    // Test 1: Basic allocation and deallocation
    size_t size = 1024;
    void* ptr = fastMalloc(size);
    assert(ptr != nullptr);
    std::cout << "  Allocated " << size << " bytes at " << ptr << std::endl;
    
    // Test 2: Write to allocated memory
    unsigned char* data = (unsigned char*)ptr;
    std::memset(data, 0xAA, size);
    for (size_t i = 0; i < size; ++i) {
        assert(data[i] == 0xAA);
    }
    std::cout << "  Memory write/read test passed!" << std::endl;
    
    // Test 3: Free memory
    fastFree(ptr);
    std::cout << "  Memory freed successfully!" << std::endl;
    
    // Test 4: Multiple allocations
    void* ptr1 = fastMalloc(512);
    void* ptr2 = fastMalloc(1024);
    void* ptr3 = fastMalloc(2048);
    assert(ptr1 != nullptr && ptr2 != nullptr && ptr3 != nullptr);
    std::cout << "  Multiple allocations successful!" << std::endl;
    
    fastFree(ptr1);
    fastFree(ptr2);
    fastFree(ptr3);
    std::cout << "  Multiple deallocations successful!" << std::endl;
}

void test_fastMalloc_with_alignPtr()
{
    std::cout << "Testing fastMalloc with alignPtr..." << std::endl;
    
    // Allocate aligned memory
    size_t alloc_size = 2048;
    void* raw_ptr = fastMalloc(alloc_size);
    assert(raw_ptr != nullptr);
    
    // Further align the pointer to 256 bytes
    unsigned char* data = (unsigned char*)raw_ptr;
    unsigned char* aligned_data = alignPtr(data, 256);
    
    std::cout << "  Raw pointer: " << (size_t)data << std::endl;
    std::cout << "  Aligned pointer (256): " << (size_t)aligned_data << std::endl;
    assert((size_t)aligned_data % 256 == 0);
    
    // Write data to aligned memory
    size_t write_size = 256;
    std::memset(aligned_data, 0x55, write_size);
    for (size_t i = 0; i < write_size; ++i) {
        assert(aligned_data[i] == 0x55);
    }
    std::cout << "  Data written to aligned memory successfully!" << std::endl;
    
    // Free original pointer
    fastFree(raw_ptr);
    std::cout << "  Memory freed successfully!" << std::endl;
}

void test_poolAllocator()
{
    std::cout << "Testing PoolAllocator..." << std::endl;
    
    PoolAllocator alloc;
    alloc.set_size_compare_ratio(0.75f);
    
    // Allocate memory from pool
    void* ptr1 = alloc.fastMalloc(512);
    void* ptr2 = alloc.fastMalloc(1024);
    assert(ptr1 != nullptr && ptr2 != nullptr);
    std::cout << "  Pool allocations successful!" << std::endl;
    
    // Free back to pool
    alloc.fastFree(ptr1);
    alloc.fastFree(ptr2);
    std::cout << "  Pool deallocations successful!" << std::endl;
    
    // Clear pool
    alloc.clear();
    std::cout << "  Pool cleared!" << std::endl;
}

int main()
{
    std::cout << "=== Starting Allocator Tests ===" << std::endl << std::endl;
    
    try {
        test_alignPtr();
        std::cout << std::endl;
        
        test_alignSize();
        std::cout << std::endl;
        
        test_fastMalloc_fastFree();
        std::cout << std::endl;
        
        test_fastMalloc_with_alignPtr();
        std::cout << std::endl;
        
        test_poolAllocator();
        std::cout << std::endl;
        
        std::cout << "=== All Tests Passed! ===" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
