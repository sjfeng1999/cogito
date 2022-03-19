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

template<typename T, template<typename> class ElementWiseOp, int BlockDimX>
struct BlockElementWise
{
    static constexpr int kBlockDimX = BlockDimX;

    using ElementWiseOpT = ElementWiseOp<T>;

    COGITO_DEVICE
    void operator()(T* input, T* output, int size){
        int tid = threadIdx.x;
        int ctaid = blockIdx.x;
        int offset = ctaid * kBlockDimX + tid;

        if (offset < size){
            ElementWiseOpT op;
            op(input + offset, output + offset);
        }
    } 

    COGITO_DEVICE
    void operator()(T* input, T* output, const T& operand, int size){
        int tid = threadIdx.x;
        int ctaid = blockIdx.x;
        int offset = ctaid * kBlockDimX + tid;

        if (offset < size){
            ElementWiseOpT op;
            op(input + offset, output + offset, operand);
        }
    } 
};

} // namespace detail
} // namespace general
} // namespace cogito