//
// 
//
//

#pragma once

#include "cogito/blas/transpose/block_level.cuh"

namespace cogito::blas {

///////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

template<typename T, int BlockDimX, int AlignSize>
COGITO_KERNEL
void TransposeKernel(int row, int col, T* A, int lda, T* B, int ldb) {
    using BlockTransposeT = BlockTranspose<T, BlockDimX, AlignSize>;

    int ctaid = blockIdx.x;

    T* block_A = A + ctaid * BlockTransposeT::kTileHeight * lda;
    T* block_B = B + ctaid * BlockTransposeT::kTileHeight;

    BlockTransposeT transpose_op;
    transpose_op(row, col, block_A, lda, block_B, ldb);
}

} // namespace detail

template<typename T>
struct Transpose {
public:
    static constexpr int kBlockDimX = 256;

public:
    Status operator()(int row, int col, T* A, int lda, T* B, int ldb, cudaStream_t stream = nullptr) {
        dim3 gDim(row / 8);
        dim3 bDim(kBlockDimX);

        Status status = Status::kSuccess;

        if (lda % 4 == 0 and ldb % 4 == 0) {
            detail::TransposeKernel<T, kBlockDimX, 4><<<gDim, bDim, 0, stream>>>(row, col, A, lda, B, ldb);
        } else {
            status = Status::kUnimplemented;
        }
        return status;
    }
};

} // namespace cogito::blas