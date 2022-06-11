//
//
//
//

#pragma once

#include "cogito/common/ldst.cuh"
#include "cogito/general/elementwise/thread_level.cuh"

namespace cogito {
namespace general {
namespace detail {

///////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, template<typename> class ElementWiseOp, int BlockDimX, int blockSize, int stripSize = 1>
struct BlockElementWise {
public:
    static constexpr int kElementSize    = sizeof(T);
    static constexpr int kBlockDimX      = BlockDimX;
    static constexpr int kBlockSize      = blockSize;
    static constexpr int kStripSize      = stripSize;
    static constexpr int kItemsPerThread = kBlockSize * kStripSize;
    static constexpr int kBlockWorkload  = kBlockDimX * kItemsPerThread;
    static constexpr LoadPolicy  kLdPolicy = LoadPolicy::kCA;
    static constexpr StorePolicy kStPolicy = StorePolicy::kWT;
    using ThreadElementWiseOpT = ThreadElementWise<T, ElementWiseOp, kItemsPerThread>;
    using ShapedTensorT        = ShapedTensor<T, kItemsPerThread>;

public:
    COGITO_DEVICE
    void operator()(const T* input, T* output, const int size){
        int tid = threadIdx.x;
        int ctaid = blockIdx.x;
        int offset = (ctaid * kBlockDimX + tid) * kItemsPerThread;

        ShapedTensorT tensor;
        // TODO (strip condition)
        if (offset < size) {
            ThreadLd<T, kLdPolicy>::load(tensor, 
                ptx::computeEffectiveAddr<mp::Log2<kElementSize>::value>(input, offset));
        }
        {
            ThreadElementWiseOpT thread_op;
            thread_op(tensor, tensor);
        }
        if (offset < size) {
            ThreadSt<T, kStPolicy>::store(tensor, 
                ptx::computeEffectiveAddr<mp::Log2<kElementSize>::value>(output, offset));
        }
    } 

    COGITO_DEVICE
    void operator()(const T* input, T* output, const T& operand, const int size){
        int tid = threadIdx.x;
        int ctaid = blockIdx.x;
        int offset = (ctaid * kBlockDimX + tid) * kItemsPerThread;

        ShapedTensorT tensor;
        // TODO (strip condition)
        if (offset < size) {
            ThreadLd<T, kLdPolicy>::load(tensor, input + offset);
        }
        {
            ThreadElementWiseOpT thread_op;
            thread_op(tensor, tensor, operand);
        }
        if (offset < size) {
            ThreadSt<T, kStPolicy>::store(tensor, output + offset);
        }
    } 
};

} // namespace detail
} // namespace general
} // namespace cogito
