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

template <typename T, template<typename> class ReduceOp, int BlockDimX, int ItemsPerThread>
struct BlockReduce {
public:
    static constexpr int kBlockDimX      = BlockDimX;
    static constexpr int kWarpNums       = kBlockDimX / kWarpSize;
    static constexpr int kItemsPerThread = ItemsPerThread;
    using ReduceOpT     = ReduceOp<T>;
    using WarpReduceT   = WarpReduce<T, ReduceOp, kItemsPerThread>;
    using ShapedTensorT = ShapedTensor<T, kItemsPerThread>;

public:
    COGITO_DEVICE 
    void operator()(T* input, T* output, int size) {

        int tid = threadIdx.x;
        int offset = tid * kItemsPerThread;

        ShapedTensorT input_tensor;
        const T identity = ReduceOpT::kIdentity;
        if (tid + offset < size) {
            ThreadLd<T>::load(input_tensor, input + offset);
        } else {
            ThreadLd<T>::load(input_tensor, identity);
        }

        T warp_res;
        {
            WarpReduceT warp_op;
            warp_res = warp_op(input_tensor);
        }

        int laneid = ptx::getLaneid();
        int warpid = tid >> 5;

        __shared__ T warp_aggregates[kWarpNums];

        if (laneid == 0){
            warp_aggregates[warpid] = warp_res;
        }
        __syncthreads();

        if (tid == 0){
            COGITO_PRAGMA_UNROLL
            for (int i = 1; i < kWarpNums; ++i){
                warp_res = thread_op(warp_res, warp_aggregates[i]);
            }
            *output = warp_res;
        }
    }
};


template <typename T, template<typename> class ReduceOp, int BlockDimX>
struct BlockAllReduce;

} // namespace detail
} // namespace general
} // namespace cogito