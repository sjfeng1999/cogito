//
// 
//
//

#pragma once

#include "cogito/general/reduce/block_level.cuh"
#include "cogito/general/reduce/thread_level.cuh"

#include <map>

namespace cogito {
namespace general {

///////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

template<typename T, template<typename> class BinaryOp, int BlockDimX>
COGITO_KERNEL
void ReduceSingleKernel(T* input, T* output, int size) {
    using BlockReduceT = BlockReduce<T, BinaryOp, BlockDimX, 1>;

    int ctaid = blockIdx.x;
    int block_offset = ctaid * BlockDimX;

    T* block_input = input + block_offset;
    T* block_output = output + ctaid;

    BlockReduceT op;
    op(block_input, block_output, size);
}

template<typename T, template<typename> class BinaryOp>
COGITO_KERNEL
void ReduceFinalKernel(T* input, T* output, int size) {
    using BinaryOpT = BinaryOp<T>;

    T val = input[0];

    BinaryOpT op;
    COGITO_PRAGMA_UNROLL
    for (int i = 1; i < size; ++i){
        val = op(val, input[i]);
    }
    *output = val;
}

} // namsespace detail

///////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, template<typename> class ReduceOp>
struct Reduce {
public:
    static constexpr int kBlockDimX = 256;
    static constexpr int kVecLength = 1;
    static constexpr int kBlockWorkload = kVecLength * kBlockDimX;

public:
    cudaError_t operator()(T* input, T* output, int size, cudaStream_t stream = nullptr){
        int gridDimX = UPPER_DIV(size, kBlockWorkload);
        
        dim3 gridDim(gridDimX, 1, 1);
        dim3 blockDim(kBlockDimX, 1, 1);

        if (gridDimX == 1) {
            auto func = detail::ReduceSingleKernel<T, ReduceOp, kBlockDimX>;
            func<<<gridDim, blockDim, 0, stream>>>(input, output, size);
        
        } else {
            T* global_workspace;
            cudaMalloc(&global_workspace, sizeof(T) * gridDimX);

            detail::ReduceSingleKernel<T, ReduceOp, kBlockDimX><<<gridDim, blockDim, 0, stream>>>(input, global_workspace, size);
            detail::ReduceFinalKernel<T, ReduceOp><<<1, 1>>>(global_workspace, output, gridDimX);

            cudaFree(global_workspace);
        }
        return cudaPeekAtLastError();
    }
};


} // namespace general
} // namespace cogito