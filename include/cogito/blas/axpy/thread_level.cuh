//
// 
//
//

#pragma once 

#include "cuda_fp16.h"
#include "cogito/tensor.cuh"

namespace cogito {
namespace blas {
namespace detail {

///////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, int Size>
struct ThreadScale {
public:
    COGITO_DEVICE
    void operator()(const T& alpha, ShapedTensor<T, Size>& A) {
        COGITO_PRAGMA_UNROLL
        for (int i = 0; i < Size; ++i) {
            A[i] *= alpha;
        }
    }
};

template<typename T, int Size>
struct ThreadFma {
public:
    COGITO_DEVICE
    void operator()(ShapedTensor<T, Size>& D, const ShapedTensor<T, Size>& A, const ShapedTensor<T, Size>& B, const ShapedTensor<T, Size>& C) {
        COGITO_PRAGMA_UNROLL
        for (int i = 0; i < Size; ++i) {
            D[i] = A[i] * B[i] + C[i];
        }
    }
};


} // namespace detail
} // namespace blas
} // namespace cogito

