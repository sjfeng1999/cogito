//
//
//
//

#pragma once

#include "cogito/ptx.h"
#include "cogito/general/reduce/warp_level.h"

namespace cogito {
namespace general {
namespace detail {

///////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, template<typename> class ReduceOp, bool Single>
struct BlockReduce {};


template <typename T, template<typename> class ReduceOp, int BlockDimX, bool Single>
struct BlockReduce<T, ReduceOp, true>
{   
public:
    static constexpr int kBlockDimX = BlockDimX;
    static constexpr int kWarpSize = kBlockDimX / kWarpSize;
private:
    __shared__ T warp_aggregates[];
public:
    BlockReduce(){
        
    }

    COGITO_DEVICE 
    void operator()(T* input, T* output, int size){
        int tid = threadIdx.x;
        int laneid = cogito::utils::get_laneid();
        int warpid = cogito::utils::get_warpid();
        T val = array[tid];

        T warp_res = WarpReduce<T, ReduceOp>(val);
        if (laneid == 0){
            warp_aggregates[warpid] = val;
        }
        __syncthreads();
        if (tid == 0){
            COGITO_UNROLL
            for (int i = 1; i < kWarpSize; ++i){
                val = ReduceOp<T>()(val, warp_aggregates[i]);
            }
            *output = val;
        }
    }
};


} // namespace detail
} // namespace general
} // namespace cogito