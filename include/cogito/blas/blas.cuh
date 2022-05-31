//
//
//
//

#pragma once 

namespace cogito {
namespace blas {

///////////////////////////////////////////////////////////////////////////////////////////////

enum class MmaType {
    kLegacy,
    kTensorCore,
};

template<typename T, MmaType type>
struct Gemm;

} // namespace blas
} // namespace cogito

#include "cogito/blas/gemm/gemm.cuh"
