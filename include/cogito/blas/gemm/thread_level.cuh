//
// 
//
//

#pragma once 

#include "cogito/tensor.cuh"

namespace cogito {
namespace blas {
namespace detail {

///////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, typename Shape>
struct ThreadMma {
public:
    static constexpr int kM = Shape::kM;
    static constexpr int kN = Shape::kN;
    static constexpr int kK = Shape::kK;
    using ShapedTensorA = ShapedTensor<T, kM * kK>;
    using ShapedTensorB = ShapedTensor<T, kN * kK>;
    using ShapedTensorC = ShapedTensor<T, kM * kN>;

public:
    COGITO_DEVICE
    void operator()(const ShapedTensorA& A, const ShapedTensorB& B, ShapedTensorC& C) {
        COGITO_PRAGMA_UNROLL
        for (int i = 0; i < kM; ++i) {
            COGITO_PRAGMA_UNROLL
            for (int j = 0; j < kN; ++j) {
                COGITO_PRAGMA_UNROLL
                for (int k = 0; k < kK; ++k) {
                    if (i % 2 == 0) {
                        C[i * kN + kN - j - 1] += A[i * kK + k] * B[k * kN + kN - j - 1];
                    } else {
                        C[i * kN + j] += A[i * kK + k] * B[k * kN + j];
                    }
                }
            }
        }
    }
};


template<typename T, typename Shape>
struct ThreadMul {
public:
    static constexpr int kM = Shape::kM;
    static constexpr int kN = Shape::kN;
    static constexpr int kK = Shape::kK;
    using ShapedTensorA = ShapedTensor<T, kM * kK>;
    using ShapedTensorB = ShapedTensor<T, kN * kK>;
    using ShapedTensorC = ShapedTensor<T, kM * kN>;

public:
    COGITO_DEVICE
    void operator()(const ShapedTensorA& A, const ShapedTensorB& B, ShapedTensorC& C) {
        COGITO_PRAGMA_UNROLL
        for (int i = 0; i < kM; ++i) {
            COGITO_PRAGMA_UNROLL
            for (int j = 0; j < kN; ++j) {
                COGITO_PRAGMA_UNROLL
                for (int k = 0; k < kK; ++k) {
                    if (i % 2 == 0) {
                        C[i * kN + kN - j - 1] = A[i * kK + k] * B[k * kN + kN - j - 1];
                    } else {
                        C[i * kN + j] = A[i * kK + k] * B[k * kN + j];
                    }
                }
            }
        }
    }
};

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

} // namespace detail
} // namespace blas
} // namespace cogito
