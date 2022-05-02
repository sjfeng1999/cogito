//
// 
//
//

#pragma once

#include "cogito/cogito.cuh"
#include "cogito/blas/blas.cuh"
#include "cogito/blas/gemm/block_level.cuh"

namespace cogito {
namespace blas {

///////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

template<typename T, MmaType type>
struct GemmConfig {};

template<>
struct GemmConfig<float, MmaType::kLegacy>{
    static constexpr int kBlockTileWidth = 128;
    static constexpr int kBlockDimX = 256;
    static constexpr MmaType kMmaType = MmaType::kLegacy;

    using type = float;
};


template<typename GemmConfig, typename T = typename GemmConfig::type>
COGITO_GLOBAL
void GemmKernel(int m, int n, int k, T alpha, T* A, int lda, T* B, int ldb, T beta, T* C, int ldc) {

    using BlockMmaT = BlockMma<typename GemmConfig::type, GemmConfig::kMmaType>;
    using TileSrcAIteratorT = typename BlockMmaT::TileSrcAIteratorT;  
    using TileSrcBIteratorT = typename BlockMmaT::TileSrcBIteratorT;  
    using TileResIteratorT  = typename BlockMmaT::TileResIteratorT;  

    BlockMmaT op;
    TileSrcAIteratorT iter_a(A, lda);
    TileSrcBIteratorT iter_b(B, ldb);
    TileResIteratorT iter_acc(C, ldc);
    TileResIteratorT iter_c(C, ldc);

    op(iter_acc, alpha, iter_a, iter_b, beta, iter_c);
}


} // namsespace detail

///////////////////////////////////////////////////////////////////////////////////////////////


template<typename T, MmaType type>
struct Gemm {
    cudaError_t operator()(int m, int n, int k, T alpha, T* A, int lda, T* B, int ldb, T beta, T* C, int ldc, cudaStream_t stream = nullptr) {
        
        using GemmConfigT = detail::GemmConfig<T, type>;

        int gridX = UPPER_DIV(m, GemmConfigT::kBlockTileWidth);
        int gridY = UPPER_DIV(n, GemmConfigT::kBlockTileWidth);

        dim3 bDim(GemmConfigT::kBlockDimX);
        dim3 gDim(gridX, gridY);

        detail::GemmKernel<GemmConfigT><<<gDim, bDim, 0, stream>>>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);

        return cudaPeekAtLastError();
    }
};


} // namespace general
} // namespace cogito