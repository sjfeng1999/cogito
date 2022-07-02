//
// 
//
//

#pragma once

#include "cogito/blas/gemm/block_level.cuh"

namespace cogito::blas {

///////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

template<int M, int N, int K>
struct GemmShape {
public:
    static constexpr int kM = M;
    static constexpr int kN = N;
    static constexpr int kK = K;
};

struct Matrix_a {};
struct Matrix_b {};

template<typename T, MmaType type>
struct GemmConfig {};

template<>
struct GemmConfig<float, MmaType::kLegacy> {
public:
    static constexpr int kBlockDimX       = 256;
    static constexpr int kBlockTileWidth  = 128;
    static constexpr int kBlockTileHeight = 128;
    static constexpr int kSplitKSize      = 1024;
    static constexpr MmaType kMmaType     = MmaType::kLegacy;
    using type = float;
};

template<typename GemmConfig, typename T = typename GemmConfig::type>
COGITO_GLOBAL 
void GemmKernel(int m, int n, int k, T alpha, T* A, int lda, T* B, int ldb, T beta, T* C, int ldc) {
    using BlockMmaT = BlockMma<typename GemmConfig::type, GemmConfig::kMmaType>;
    using TileSrcAIteratorT = typename BlockMmaT::TileSrcAIteratorT;  
    using TileSrcBIteratorT = typename BlockMmaT::TileSrcBIteratorT;  
    using TileResIteratorT  = typename BlockMmaT::TileResIteratorT;  

    __shared__ T shared_a[TileSrcAIteratorT::kSharedSize];
    __shared__ T shared_b[TileSrcBIteratorT::kSharedSize];

    int ctaid_x = blockIdx.x;
    int ctaid_y = blockIdx.y;

    T* block_A = A + ctaid_y * GemmConfig::kBlockTileHeight * lda;
    T* block_B = B + ctaid_x * GemmConfig::kBlockTileWidth;
    T* block_C = C + ctaid_y * GemmConfig::kBlockTileHeight * ldc + ctaid_x * GemmConfig::kBlockTileWidth;
    
    TileSrcAIteratorT iter_a(block_A, lda, lda, shared_a);
    TileSrcBIteratorT iter_b(block_B, ldb, ldb, shared_b);
    TileResIteratorT iter_c(block_C, ldc);
    __syncthreads();

    BlockMmaT op;
    op(alpha, iter_a, iter_b, beta, iter_c);
}


template<typename GemmConfig, typename T = typename GemmConfig::type>
COGITO_GLOBAL 
void GemmSplitK_Kernel(int m, int n, int k, T alpha, T* A, int lda, T* B, int ldb, T beta, T* C, int ldc);

} // namsespace detail

///////////////////////////////////////////////////////////////////////////////////////////////


template<typename T, MmaType type>
struct Gemm {
public:
    using GemmConfigT = detail::GemmConfig<T, type>;

public:
    cudaError_t operator()(int m, int n, int k, T alpha, T* A, int lda, T* B, int ldb, T beta, T* C, int ldc, cudaStream_t stream = nullptr) {

        int gridX = UPPER_DIV(m, GemmConfigT::kBlockTileWidth);
        int gridY = UPPER_DIV(n, GemmConfigT::kBlockTileWidth);

        dim3 bDim(GemmConfigT::kBlockDimX);
        dim3 gDim(gridX, gridY);

        detail::GemmKernel<GemmConfigT><<<gDim, bDim, 0, stream>>>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);

        return cudaPeekAtLastError();
    }
};

} // namespace cogito::blas
