//
//
//
//

#pragma once 

#include "cogito/cogito.cuh"
#include "cogito/ptx.cuh"

#include "cogito/general/general.cuh"
#include "cogito/general/elementwise/block_level.cuh"

namespace cogito {
namespace general {

///////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

template<typename T, template<typename> class ElementWiseOp, int BlockDimX, int AlignSize = 1>
COGITO_KERNEL
void ElementWiseKernel(const T* input, T* output, const int size) {
    using BlockElementWiseFullT = BlockElementWise<T, ElementWiseOp, BlockDimX, 4, 1, true>;
    using BlockElementWiseTailT = BlockElementWise<T, ElementWiseOp, BlockDimX, AlignSize, 1, false>;

    int ctaid = blockIdx.x;
    int block_offset = ctaid * BlockElementWiseFullT::kWorkloadLine;

    const T* block_input = input + block_offset;
    T* block_output = output + block_offset;

    if (ctaid == blockDim.x - 1) {
        int tail = size % BlockElementWiseFullT::kWorkloadLine;
        BlockElementWiseTailT block_op;
        block_op(block_input, block_output, tail);
    } else {
        BlockElementWiseFullT block_op;
        block_op(block_input, block_output, BlockElementWiseFullT::kWorkloadLine);
    }
}


template<typename T, template<typename> class ElementWiseOp, int BlockDimX, int AlignSize = 1>
COGITO_KERNEL
void ElementWiseKernel(const T* input, T* output, const T operand, const int size) {
    using BlockElementWiseFullT = BlockElementWise<T, ElementWiseOp, BlockDimX, 4, 1, true>;
    using BlockElementWiseTailT = BlockElementWise<T, ElementWiseOp, BlockDimX, AlignSize, 1, false>;

    int ctaid = blockIdx.x;
    int block_offset = ctaid * BlockElementWiseFullT::kWorkloadLine;

    const T* block_input = input + block_offset;
    T* block_output = output + block_offset;

    if (ctaid == blockDim.x - 1) {
        int tail = size % BlockElementWiseFullT::kWorkloadLine;
        BlockElementWiseTailT block_op;
        block_op(block_input, block_output, operand, tail);
    } else {
        BlockElementWiseFullT block_op;
        block_op(block_input, block_output, operand, BlockElementWiseFullT::kWorkloadLine);
    }
}

} // namsespace detail

///////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, template<typename> class ElementWiseOp>
struct ElementWise {
public:
    static constexpr int kBlockDimX = 256;
    
public:
    Status operator()(T* input, T* output, int size, cudaStream_t stream = nullptr) {
        dim3 bDim(kBlockDimX);
        dim3 gDim(UPPER_DIV(size, kBlockDimX * 4));

        if (size % 4 == 0) {
            detail::ElementWiseKernel<T, ElementWiseOp, kBlockDimX, 4><<<gDim, bDim, 0, stream>>>(input, output, size);
        } else if (size % 2 == 0) {
            detail::ElementWiseKernel<T, ElementWiseOp, kBlockDimX, 2><<<gDim, bDim, 0, stream>>>(input, output, size);
        } else {
            detail::ElementWiseKernel<T, ElementWiseOp, kBlockDimX, 1><<<gDim, bDim, 0, stream>>>(input, output, size);
        }
        return Status::kSuccess;
    }

    Status operator()(T* input, T* output, const T operand, int size, cudaStream_t stream = nullptr) {
        dim3 bDim(kBlockDimX);
        dim3 gDim(UPPER_DIV(size, kBlockDimX * 4));

        if (size % 4 == 0) {
            detail::ElementWiseKernel<T, ElementWiseOp, kBlockDimX, 4><<<gDim, bDim, 0, stream>>>(input, output, operand, size);
        } else if (size % 2 == 0) {
            detail::ElementWiseKernel<T, ElementWiseOp, kBlockDimX, 2><<<gDim, bDim, 0, stream>>>(input, output, operand, size);
        } else {
            detail::ElementWiseKernel<T, ElementWiseOp, kBlockDimX, 1><<<gDim, bDim, 0, stream>>>(input, output, operand, size);
        }
        return Status::kSuccess;
    }
};


} // namespace general
} // namespace cogito
