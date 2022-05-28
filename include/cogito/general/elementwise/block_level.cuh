//
//
//
//

#pragma once

#include "cogito/cogito.cuh"
#include "cogito/common/ldst.cuh"
#include "cogito/general/elementwise/thread_level.cuh"

namespace cogito {
namespace general {
namespace detail {

///////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, template<typename> class ElementWiseOp, int BlockDimX, int ItemPerThread>
struct BlockElementWise {
public:
    static constexpr int kBlockDimX     = BlockDimX;
    static constexpr int kItemPerThread = ItemPerThread;
    static constexpr int kBlockWorkload = kBlockDimX * kItemPerThread;

    using ThreadElementWiseOpT = ThreadElementWise<T, ElementWiseOp, kItemPerThread>;

public:
    COGITO_DEVICE
    void operator()(const T* input, T* output, const int size){
        int tid = threadIdx.x;
        int ctaid = blockIdx.x;
        int offset = (ctaid * kBlockDimX + tid) * kItemPerThread;

        ShapedTensor<T, kItemPerThread> tensor;
        ThreadLoad<T, kItemPerThread>::load(tensor, input + offset, static_cast<bool>(offset < size));
        {
            ThreadElementWiseOpT op;
            op(tensor, tensor);
        }
        ThreadStore<T, kItemPerThread>::store(tensor, output + offset, static_cast<bool>(offset < size));
    } 

    COGITO_DEVICE
    void operator()(const T* input, T* output, const T& operand, const int size){
        int tid = threadIdx.x;
        int ctaid = blockIdx.x;
        int offset = (ctaid * kBlockDimX + tid) * kItemPerThread;

        if (offset < size){
            ThreadElementWiseOpT op;
            op(input + offset, output + offset, operand);
        }
    } 
};

} // namespace detail
} // namespace general
} // namespace cogito