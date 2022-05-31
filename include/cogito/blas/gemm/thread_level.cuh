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
    static constexpr int kK = 4;
    using ShapedTensorA = ShapedTensor<T, kM>;
    using ShapedTensorB = ShapedTensor<T, kN>;
    using ShapedTensorC = ShapedTensor<T, kM * kN>;

public:
    COGITO_DEVICE
    void operator()(const ShapedTensorA& A, const ShapedTensorB& B, ShapedTensorC& C){
        COGITO_PRAGMA_UNROLL
        for (int i = 0; i < kM; ++i){
            COGITO_PRAGMA_UNROLL
            for (int j = 0; j < kN; ++j){
                C[i * kM + j] += A[i] * B[j];
            }
        }
    }
};


} // namespace detail
} // namespace blas
} // namespace cogito
