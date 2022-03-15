//
//
//
//

#pragma once

#include "cogito/cogito.cuh"

namespace cogito {
namespace general {
namespace detail {

///////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, template<typename> class UnaryOp, int BlockDimX>
struct BlockElementWise
{
    static constexpr int kBlockDimX = BlockDimX;

    COGITO_DEVICE
    void operator()(T* input, T* output, int size){
        int tid = threadIdx.x;
        int ctaid = blockIdx.x;
        int offset = ctaid * kBlockDimX + tid;

        if (offset < size){
            UnaryOp<T> op;
            
            op(input + offset, output + offset);
        }
    } 
};

} // namespace detail
} // namespace general
} // namespace cogito