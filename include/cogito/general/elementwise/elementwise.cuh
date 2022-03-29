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


template<typename T, template<typename> class ElementWiseOp, int BlockDimX>
COGITO_GLOBAL
void ElementWiseKernel(T* input, T* output, int size){

    using BlockElementWiseT = BlockElementWise<T, ElementWiseOp, BlockDimX>;

    BlockElementWiseT op;
    op(input, output, size);
}


template<typename T, template<typename> class ElementWiseOp, int BlockDimX>
COGITO_GLOBAL
void ElementWiseKernel(T* input, T* output, const T operand, int size){

    using BlockElementWiseT = BlockElementWise<T, ElementWiseOp, BlockDimX>;

    BlockElementWiseT op;
    op(input, output, operand, size);
}

template<typename T, template<typename> class ElementWiseOp, int BlockDimX>
COGITO_GLOBAL
void ElementWiseKernel(T* input, T* output, T* operand, int size){

    using BlockElementWiseT = BlockElementWise<T, ElementWiseOp, BlockDimX>;

    BlockElementWiseT op;
    op(input, output, *operand, size);
}


} // namsespace detail

///////////////////////////////////////////////////////////////////////////////////////////////


template<typename T, template<typename> class ElementWiseOp>
struct ElementWise
{   
    static constexpr int kBlockDimX = 256;
    static constexpr int kVecLength = 1;
    static constexpr int kBlockWorkload = kVecLength * kBlockDimX;
    

    cudaError_t operator()(T* input, T* output, int size, cudaStream_t stream = nullptr){
        int gridDimX = UPPER_DIV(size, kBlockWorkload);
        
        dim3 gridDim(gridDimX, 1, 1);
        dim3 blockDim(kBlockDimX, 1, 1);

        detail::ElementWiseKernel<T, ElementWiseOp, kBlockDimX><<<gridDim, blockDim, 0, stream>>>(input, output, size);
        return cudaPeekAtLastError();
    }


    // operand is Host-Value
    cudaError_t operator()(T* input, T* output, const T operand, int size, cudaStream_t stream = nullptr){
        int gridDimX = UPPER_DIV(size, kBlockWorkload);
        
        dim3 gridDim(gridDimX, 1, 1);
        dim3 blockDim(kBlockDimX, 1, 1);

        detail::ElementWiseKernel<T, ElementWiseOp, kBlockDimX><<<gridDim, blockDim, 0, stream>>>(input, output, operand, size);
        return cudaPeekAtLastError();
    }

    // operand is Device-Pointer
    cudaError_t operator()(T* input, T* output, T* operand, int size, cudaStream_t stream = nullptr){
        int gridDimX = UPPER_DIV(size, kBlockWorkload);
        
        dim3 gridDim(gridDimX, 1, 1);
        dim3 blockDim(kBlockDimX, 1, 1);

        detail::ElementWiseKernel<T, ElementWiseOp, kBlockDimX><<<gridDim, blockDim, 0, stream>>>(input, output, operand, size);
        return cudaPeekAtLastError();
    }

};


} // namespace general
} // namespace cogito