//
//
//
//

#pragma once 

#include "cogito/cogito.cuh"
#include "cogito/ptx.cuh"

#include "cogito/general/elementwise/block_level.cuh"

namespace cogito {
namespace general {

///////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, template<typename> class UnaryOp, int BlockDimX>
COGITO_GLOBAL
void ElementWiseKernel(T* input, T* output, int size){

    using BlockElementWiseT = detail::BlockElementWise<T, UnaryOp, BlockDimX>;

    BlockElementWiseT()(input, output, size);
}


template<typename T, template<typename> class UnaryOp>
struct ElementWise
{   
    static constexpr int kBlockDimX = 256;
    static constexpr int kVecLength = 1;
    static constexpr int kBlockWorkload = kVecLength * kBlockDimX;
    

    cudaError_t operator()(T* input, T* output, int size, cudaStream_t stream = 0){
        int gridDimX = UPPER_DIV(size, kBlockWorkload);
        
        dim3 gridDim(gridDimX, 1, 1);
        dim3 blockDim(kBlockDimX, 1, 1);

        auto func = ElementWiseKernel<T, UnaryOp, kBlockDimX>;
        func<<<gridDim, blockDim, 0, stream>>>(input, output, size);
        return cudaPeekAtLastError();
    }
};


} // namespace general
} // namespace cogito