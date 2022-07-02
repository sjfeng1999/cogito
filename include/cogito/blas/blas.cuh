//
//
//
//

#pragma once 

namespace cogito::blas {

///////////////////////////////////////////////////////////////////////////////////////////////

enum class MmaType {
    kLegacy,
    kTensorCore,
};

template<typename T, MmaType type>
struct Gemm;

template<typename T, MmaType type>
struct GemmSplitK;

} // namespace cogito::blas

#include "cogito/blas/gemm/gemm.cuh"
