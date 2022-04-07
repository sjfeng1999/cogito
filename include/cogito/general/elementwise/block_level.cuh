//
//
//
//

#pragma once

#include "cogito/cogito.cuh"
#include "cogito/general/elementwise/thread_level.cuh"

namespace cogito {
namespace general {
namespace detail {

///////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, template<typename> class ElementWiseOp, int BlockDimX>
struct BlockElementWise {

    static constexpr int kBlockDimX = BlockDimX;
    static constexpr int kVecLength = 1;
    
    using ThreadElementWiseOpT = ThreadElementWise<T, ElementWiseOp, kVecLength>;

    COGITO_DEVICE
    void operator()(T* input, T* output, int size){
        int tid = threadIdx.x;
        int ctaid = blockIdx.x;
        int offset = (ctaid * kBlockDimX + tid) * kVecLength;

        if (offset < size){
            ThreadElementWiseOpT op;
            op(input + offset, output + offset);
        }
    } 

    COGITO_DEVICE
    void operator()(T* input, T* output, const T& operand, int size){
        int tid = threadIdx.x;
        int ctaid = blockIdx.x;
        int offset = ctaid * kBlockDimX + tid;

        if (offset < size){
            ThreadElementWiseOpT op;
            op(input + offset, output + offset, operand);
        }
    } 
};

} // namespace detail
} // namespace general
} // namespace cogito