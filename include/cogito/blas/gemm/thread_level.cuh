//
// 
//
//

#pragma once 

#include "cogito/cogito.cuh"
#include "cogito/tensor.cuh"

namespace cogito {
namespace blas {
namespace detail {

///////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, MmaType type = MmaType::kLegacy>
struct ThreadMma {
public:
    static constexpr int kM = 4;
    static constexpr int kN = 4;
    static constexpr int kK = 1;
    using ShapedTensorA = ShapedTensor<T, kM * kK>;
    using ShapedTensorB = ShapedTensor<T, kN * kK>;
    using ShapedTensorC = ShapedTensor<T, kM * kN>;

public:
    COGITO_DEVICE
    void operator()(const T alpha, const ShapedTensorA& A, const ShapedTensorB& B, const T beta, ShapedTensorC& C) {
        COGITO_PRAGMA_UNROLL
        for (int i = 0; i < kM; ++i) {
            COGITO_PRAGMA_UNROLL
            for (int j = 0; j < kN; ++j) {
                COGITO_PRAGMA_UNROLL
                for (int k = 0; k < kK; ++k) {
                    if (i % 2 == 0) {
                        C[i * kN + kN - j - 1] += alpha * A[i * kK + k] * B[k * kK + kN - j - 1];
                    } else {
                        C[i * kN + j] += alpha * A[i * kK + k] * B[k * kK + j];
                    }
                }
            }
        }
    }
};


} // namespace detail
} // namespace blas
} // namespace cogito
