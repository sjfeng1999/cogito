//
//
//
//

#pragma once

#include "cogito/ptx.cuh"
#include "cogito/general/reduce/warp_level.cuh"

namespace cogito {
namespace general {
namespace detail {

///////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, template<typename> class ReduceOp, int BlockDimX>
struct BlockReduce {

    static constexpr int kBlockDimX = BlockDimX;
    static constexpr int kWarpNums  = kBlockDimX / kWarpSize;

    using ReduceOpT   = ReduceOp<T>;
    using WarpReduceT = WarpReduce<T, ReduceOp>;

    COGITO_DEVICE 
    void operator()(T* input, T* output, int size){

        int tid = threadIdx.x;
        int ctaid = blockIdx.x;
        int block_offset = ctaid * kBlockDimX;

        __shared__ T warp_aggregates[kWarpNums];

        int laneid = cogito::utils::getLaneid();
        int warpid = tid >> 5;

        T val;
        if (tid + block_offset < size) {
            val = input[tid + block_offset];
        } else {
            val = ReduceOpT::kIdentity;
        }

        WarpReduceT warp_op;
        T warp_res = warp_op(&val);

        if (laneid == 0){
            warp_aggregates[warpid] = warp_res;
        }

        __syncthreads();
        if (tid == 0){
            ReduceOpT op;

            COGITO_PRAGMA_UNROLL
            for (int i = 1; i < kWarpNums; ++i){
                warp_res = op(&warp_res, &warp_aggregates[i]);
            }
            output[ctaid] = warp_res;
        }
    }
};


} // namespace detail
} // namespace general
} // namespace cogito