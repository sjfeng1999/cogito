//
//
//
//

#pragma once

#include "cogito/common/ldst.cuh"
#include "cogito/general/elementwise/thread_level.cuh"

namespace cogito::general::detail {

///////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, template<typename> class ElementWiseOp, int BlockDimX, int blockSize, int stripSize, bool Full>
struct BlockElementWise {
public:
    static constexpr bool kFull        = Full;
    static constexpr int kBlockDimX    = BlockDimX;
    static constexpr int kBlockSize    = blockSize;
    static constexpr int kStripSize    = stripSize;
    static constexpr int kWorkloadLine = kBlockDimX * kBlockSize * kStripSize;
    static constexpr LoadPolicy  kLdPolicy = LoadPolicy::kCS;
    static constexpr StorePolicy kStPolicy = StorePolicy::kWT;
    using ShapedTensorT        = ShapedTensor<T, kBlockSize>;
    using ThreadElementWiseOpT = ThreadElementWise<T, ElementWiseOp, kBlockSize>;

public:
    COGITO_DEVICE
    void operator()(const T* input, T* output, const int size) {
        int tid = threadIdx.x;
        int offset = tid * kBlockSize;

        ShapedTensorT tensor;
        ThreadElementWiseOpT thread_op;

        COGITO_PRAGMA_UNROLL
        for (int i = 0; i < kStripSize; ++i) {
            if constexpr (kFull) {
                ThreadLd<T, kLdPolicy>::load(tensor, input + offset);
                thread_op(tensor, tensor);
                ThreadSt<T, kStPolicy>::store(tensor, output + offset);
            } else if (offset < size) {
                ThreadLd<T, kLdPolicy>::load(tensor, input + offset);
                thread_op(tensor, tensor);
                ThreadSt<T, kStPolicy>::store(tensor, output + offset);
            }
            offset += kWorkloadLine;
        }
    } 

    COGITO_DEVICE
    void operator()(const T* input, T* output, const T operand, const int size) {
        int tid = threadIdx.x;
        int offset = tid * kBlockSize;

        ShapedTensorT tensor;
        ThreadElementWiseOpT thread_op;

        COGITO_PRAGMA_UNROLL
        for (int i = 0; i < kStripSize; ++i) {
            if constexpr (kFull) {
                ThreadLd<T, kLdPolicy>::load(tensor, input + offset);
                thread_op(tensor, operand, tensor);
                ThreadSt<T, kStPolicy>::store(tensor, output + offset);
            } else if (offset < size) {
                ThreadLd<T, kLdPolicy>::load(tensor, input + offset);
                thread_op(tensor, operand, tensor);
                ThreadSt<T, kStPolicy>::store(tensor, output + offset);
            }
            offset += kWorkloadLine;
        }
    } 
};


template<typename T, template<typename> class ElementWiseOp, int BlockDimX, int blockSize>
struct BlockElementWiseAll {
public:
    static constexpr int kBlockDimX    = BlockDimX;
    static constexpr int kBlockSize    = blockSize;
    static constexpr int kWorkloadLine = kBlockDimX * kBlockSize;
    static constexpr LoadPolicy  kLdPolicy = LoadPolicy::kCS;
    static constexpr StorePolicy kStPolicy = StorePolicy::kWT;
    using ShapedTensorT        = ShapedTensor<T, kBlockSize>;
    using ThreadElementWiseOpT = ThreadElementWise<T, ElementWiseOp, kBlockSize>;

public:
    COGITO_DEVICE
    void operator()(const T* input, T* output, const int size) {
        int tid = threadIdx.x;
        int offset = tid * kBlockSize;

        ShapedTensorT tensor;
        ThreadElementWiseOpT thread_op;
        
        for (; offset < size; offset += kWorkloadLine) {
            ThreadLd<T, kLdPolicy>::load(tensor, input + offset) ;
            thread_op(tensor, tensor);
            ThreadSt<T, kStPolicy>::store(tensor, output + offset);
        } 
    } 

    COGITO_DEVICE
    void operator()(const T* input, T* output, const T& operand, const int size) {
        int tid = threadIdx.x;
        int offset = tid * kBlockSize;

        ShapedTensorT tensor;
        ThreadElementWiseOpT thread_op;

        for (; offset < size; offset += kWorkloadLine) {
            ThreadLd<T, kLdPolicy>::load(tensor, input + offset);
            thread_op(tensor, operand, tensor);
            ThreadSt<T, kStPolicy>::store(tensor, output + offset);
        };
    } 
};


template<typename T, template<typename> class ElementWiseOp, int BlockDimX, int blockSize>
struct BlockElementWiseTwice;

} // namespace cogito
