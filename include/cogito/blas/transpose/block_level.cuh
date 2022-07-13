//
//
//
//

#pragma once

#include "cogito/common/ldst.cuh"

namespace cogito::blas::detail {

template<typename T, int BlockDimX, int BlockSize>
struct BlockTranspose {
public:
    static constexpr int kBlockDimX  = BlockDimX;
    static constexpr int kBlockSize  = BlockSize;
    static constexpr int kTileHeight = 8;
    static constexpr int kTileWidth  = 128;
    static constexpr int kSharedSize = kTileWidth * kTileHeight;
    using ShapedTensorT = ShapedTensor<T, kBlockSize>;

public:
    COGITO_DEVICE
    void operator()(const int row, const int column, const T* A, const int lda, T* B, const int ldb) {
        int tid = threadIdx.x;
        int offset_A, offset_B;
        {
            int warpid = tid / kWarpSize;
            int laneid = tid % kWarpSize;
            offset_A = warpid * lda + laneid * kBlockSize;
            offset_B = warpid + laneid * kBlockSize * ldb;
        }

        ShapedTensorT tensor;

        // TODO 1.use shared memory to rearrangear  2. process condition when lda not alignas TileWidth
        for (int k = 0; k < column; k += kTileWidth) {
            ThreadLd<T>::load(tensor, A + offset_A);
            ThreadSt<T>::stripedStore<0, 1>(tensor, B + offset_B, ldb, mp::Range2Type<0, kBlockSize>{});

            offset_A += kTileWidth;
            offset_B += kTileWidth * ldb;
        }
    }
};

} // namespace cogito::blas::detail
