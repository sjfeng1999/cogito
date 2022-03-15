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
        err = static_cast<float>(array1[i] - array2[i]);
        if (err > epsilon){
            printf("Error pos:%3d  err:%.5f\n", i, err);
            return false;
        }
    }
    return true;
}

template <typename T>
void initTensor(T* input, int size){
    for (int i = 0; i < size; ++i){
        input[i] = static_cast<T>(i & 0xF);
    }
}

} // namespace test
} // namespace cogito