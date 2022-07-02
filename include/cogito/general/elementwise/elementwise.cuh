//
//
//
//

#pragma once 

#include "cogito/cogito.cuh"
#include "cogito/ptx.cuh"

#include "cogito/general/general.cuh"
#include "cogito/general/elementwise/block_level.cuh"

namespace cogito::general {

///////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

template<typename T, template<typename> class ElementwiseOp, int BlockDimX, int AlignSize = 1>
COGITO_KERNEL
void ElementwiseKernel(const T* input, T* output, const int size) {
    using BlockElementwiseFullT = BlockElementwise<T, ElementwiseOp, BlockDimX, 4, 1, true>;
    using BlockElementwiseTailT = BlockElementwise<T, ElementwiseOp, BlockDimX, AlignSize, 1, false>;

    int ctaid = blockIdx.x;
    int block_offset = ctaid * BlockElementwiseFullT::kWorkloadLine;

    const T* block_input = input + block_offset;
    T* block_output = output + block_offset;

    if (ctaid == blockDim.x - 1) {
        int tail = size % BlockElementwiseFullT::kWorkloadLine;
        BlockElementwiseTailT block_op;
        block_op(block_input, block_output, tail);
    } else {
        BlockElementwiseFullT block_op;
        block_op(block_input, block_output, BlockElementwiseFullT::kWorkloadLine);
    }
}


template<typename T, template<typename> class ElementwiseOp, int BlockDimX, int AlignSize = 1>
COGITO_KERNEL
void ElementwiseKernel(const T* input, T* output, const T operand, const int size) {
    using BlockElementwiseFullT = BlockElementwise<T, ElementwiseOp, BlockDimX, 4, 1, true>;
    using BlockElementwiseTailT = BlockElementwise<T, ElementwiseOp, BlockDimX, AlignSize, 1, false>;

    int ctaid = blockIdx.x;
    int block_offset = ctaid * BlockElementwiseFullT::kWorkloadLine;

    const T* block_input = input + block_offset;
    T* block_output = output + block_offset;

    if (ctaid == blockDim.x - 1) {
        int tail = size % BlockElementwiseFullT::kWorkloadLine;
        BlockElementwiseTailT block_op;
        block_op(block_input, block_output, operand, tail);
    } else {
        BlockElementwiseFullT block_op;
        block_op(block_input, block_output, operand, BlockElementwiseFullT::kWorkloadLine);
    }
}

} // namsespace detail

///////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, template<typename> class ElementwiseOp>
struct Elementwise {
public:
    static constexpr int kBlockDimX = 256;
    
public:
    Status operator()(T* input, T* output, int size, cudaStream_t stream = nullptr) {
        dim3 bDim(kBlockDimX);
        dim3 gDim(UPPER_DIV(size, kBlockDimX * 4));

        if (size % 4 == 0) {
            detail::ElementwiseKernel<T, ElementwiseOp, kBlockDimX, 4><<<gDim, bDim, 0, stream>>>(input, output, size);
        } else if (size % 2 == 0) {
            detail::ElementwiseKernel<T, ElementwiseOp, kBlockDimX, 2><<<gDim, bDim, 0, stream>>>(input, output, size);
        } else {
            detail::ElementwiseKernel<T, ElementwiseOp, kBlockDimX, 1><<<gDim, bDim, 0, stream>>>(input, output, size);
        }
        return Status::kSuccess;
    }

    Status operator()(T* input, T* output, const T operand, int size, cudaStream_t stream = nullptr) {
        dim3 bDim(kBlockDimX);
        dim3 gDim(UPPER_DIV(size, kBlockDimX * 4));

        if (size % 4 == 0) {
            detail::ElementwiseKernel<T, ElementwiseOp, kBlockDimX, 4><<<gDim, bDim, 0, stream>>>(input, output, operand, size);
        } else if (size % 2 == 0) {
            detail::ElementwiseKernel<T, ElementwiseOp, kBlockDimX, 2><<<gDim, bDim, 0, stream>>>(input, output, operand, size);
        } else {
            detail::ElementwiseKernel<T, ElementwiseOp, kBlockDimX, 1><<<gDim, bDim, 0, stream>>>(input, output, operand, size);
        }
        return Status::kSuccess;
    }
};

} // namespace cogito::general
