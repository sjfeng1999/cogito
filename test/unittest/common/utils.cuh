//
//
//
//

#pragma once

#include <cstdlib>
#include "cogito/cogito.cuh"

namespace cogito {
namespace test {
    
///////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
bool verifyResult(T* array1, T* array2, int size, float epsilon=5e-3f) {
    float err = 0.0f;
    for (int i = 0; i < size; ++i){
        err = abs(static_cast<float>(array1[i] - array2[i]));
        if (err > epsilon){
            printf("Error pos:%3d  left:%.4f  right:%.4f\n", i, 
                                                             static_cast<float>(array1[i]), 
                                                             static_cast<float>(array2[i]));
            return false;
        }
    }
    return true;
}

template<typename T>
void initTensor(T* input, int size){
    std::srand(time(NULL));
    for (int i = 0; i < size; ++i){
        input[i] = static_cast<T>((std::rand() % 10) - 5);
    }
}

template<typename T>
void printTensor(T* input, int size){
    for (int i = 0; i < size; ++i){
        printf("%.1f, ", input[i]);
        if (i % 32 == 32 - 1) {
            printf("\n");
        }
    }
}

} // namespace test
} // namespace cogito