#include <iostream>
#include "test/aligned_memory.h"

int main()
{
    float* data_ptr;

    data_ptr = (float*)aligned_malloc(10);
    if (data_ptr != nullptr) aligned_free((void*)data_ptr);

    std::cout << data_ptr << std::endl;
    
    return 0;
};