//
//
//
//

#pragma once

#include "cogito/cogito.cuh"

namespace cogito {
namespace test {
    
///////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
bool verifyResult(T* array1, T* array2, int size, float epsilon=1e-4f){
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

template <typename T>
void initTensor(T* input, int size){
    for (int i = 0; i < size; ++i){
        input[i] = static_cast<T>(i & 0xFF);
    }
}

} // namespace test
} // namespace cogito