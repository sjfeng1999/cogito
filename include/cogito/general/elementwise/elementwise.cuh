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

template<typename T, template<typename> class ElementWiseOp, int BlockDimX, int ItemPerThread = 1>
COGITO_KERNEL
void ElementWiseKernel(T* input, T* output, const int size) {
    using BlockElementWiseT = BlockElementWise<T, ElementWiseOp, BlockDimX, ItemPerThread>;

    int ctaid = blockIdx.x;
    int block_offset = ctaid * BlockDimX * ItemPerThread;

    T* block_input = input + block_offset;
    T* block_output = output + block_offset;

    BlockElementWiseT block_op;
    block_op(block_input, block_output, size);
}


template<typename T, template<typename> class ElementWiseOp, int BlockDimX, int ItemPerThread = 1>
COGITO_KERNEL
void ElementWiseKernel(T* input, T* output, const T operand, const int size) {
    using BlockElementWiseT = BlockElementWise<T, ElementWiseOp, BlockDimX, ItemPerThread>;

    int ctaid = blockIdx.x;
    int block_offset = ctaid * BlockDimX * ItemPerThread;

    T* block_input = input + block_offset;
    T* block_output = output + block_offset;

    BlockElementWiseT block_op;
    block_op(block_input, block_output, &operand, size);
}


template<typename T, template<typename> class ElementWiseOp, int BlockDimX, int ItemPerThread = 1>
COGITO_KERNEL
void ElementWiseKernel(T* input, T* output, T* operand, const int size) {
    using BlockElementWiseT = BlockElementWise<T, ElementWiseOp, BlockDimX, ItemPerThread>;

    int ctaid = blockIdx.x;
    int block_offset = ctaid * BlockDimX * ItemPerThread;

    T* block_input = input + block_offset;
    T* block_output = output + block_offset;

    BlockElementWiseT block_op;
    block_op(block_input, block_output, operand, size);
}

} // namsespace detail

///////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, template<typename> class ElementWiseOp>
struct ElementWise {
public:
    static constexpr int kBlockDimX = 256;
    
public:
    cudaError_t operator()(T* input, T* output, int size, cudaStream_t stream = nullptr) {
        
        dim3 blockDim(kBlockDimX, 1, 1);

        if (size % 4 == 0) {
            dim3 gridDim(UPPER_DIV(size >> 2, kBlockDimX), 1, 1);
            detail::ElementWiseKernel<T, ElementWiseOp, kBlockDimX, 4><<<gridDim, blockDim, 0, stream>>>(input, output, size);

        } else if (size % 2 == 0) {
            dim3 gridDim(UPPER_DIV(size >> 1, kBlockDimX), 1, 1);
            detail::ElementWiseKernel<T, ElementWiseOp, kBlockDimX, 2><<<gridDim, blockDim, 0, stream>>>(input, output, size);

        } else {
            dim3 gridDim(UPPER_DIV(size, kBlockDimX), 1, 1);
            detail::ElementWiseKernel<T, ElementWiseOp, kBlockDimX, 1><<<gridDim, blockDim, 0, stream>>>(input, output, size);
        }
        return cudaPeekAtLastError();
    }

    // operand is Host-Value
    cudaError_t operator()(T* input, T* output, const T operand, int size, cudaStream_t stream = nullptr) {

        dim3 blockDim(kBlockDimX, 1, 1);

        if (size % 4 == 0) {
            dim3 gridDim(UPPER_DIV(size / 4, kBlockDimX), 1, 1);
            detail::ElementWiseKernel<T, ElementWiseOp, kBlockDimX, 4><<<gridDim, blockDim, 0, stream>>>(input, output, operand, size);

        } else if (size % 2 == 0) {
            dim3 gridDim(UPPER_DIV(size / 2, kBlockDimX), 1, 1);
            detail::ElementWiseKernel<T, ElementWiseOp, kBlockDimX, 2><<<gridDim, blockDim, 0, stream>>>(input, output, operand, size);

        } else {
            dim3 gridDim(UPPER_DIV(size, kBlockDimX), 1, 1);
            detail::ElementWiseKernel<T, ElementWiseOp, kBlockDimX, 1><<<gridDim, blockDim, 0, stream>>>(input, output, operand, size);
        }
        return cudaPeekAtLastError();
    }

    // operand is Device-Pointer
    cudaError_t operator()(T* input, T* output, T* operand, int size, cudaStream_t stream = nullptr) {

        dim3 blockDim(kBlockDimX, 1, 1);

        if (size % 4 == 0) {
            dim3 gridDim(UPPER_DIV(size / 4, kBlockDimX), 1, 1);
            detail::ElementWiseKernel<T, ElementWiseOp, kBlockDimX, 4><<<gridDim, blockDim, 0, stream>>>(input, output, operand, size);

        } else if (size % 2 == 0) {
            dim3 gridDim(UPPER_DIV(size / 2, kBlockDimX), 1, 1);
            detail::ElementWiseKernel<T, ElementWiseOp, kBlockDimX, 2><<<gridDim, blockDim, 0, stream>>>(input, output, operand, size);

        } else {
            dim3 gridDim(UPPER_DIV(size, kBlockDimX), 1, 1);
            detail::ElementWiseKernel<T, ElementWiseOp, kBlockDimX, 1><<<gridDim, blockDim, 0, stream>>>(input, output, operand, size);
        }
        return cudaPeekAtLastError();
    }

};


} // namespace general
} // namespace cogito
