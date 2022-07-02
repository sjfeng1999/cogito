//
// 
//
//

#pragma once

#include "cogito/general/reduce/block_level.cuh"
#include "cogito/general/reduce/thread_level.cuh"

namespace cogito::general {

///////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

template<typename T, template<typename> class BinaryOp, int BlockDimX, int AlignSize = 1>
COGITO_KERNEL
void ReduceSingleKernel(const T* input, T* output, int size) {
    using BlockReduceFullT = BlockReduce<T, BinaryOp, BlockDimX, 4, 1, true>;
    using BlockReduceTailT = BlockReduce<T, BinaryOp, BlockDimX, AlignSize, 1, false>;

    int ctaid = blockIdx.x;
    int block_offset = ctaid * BlockReduceFullT::kWorkloadLine;

    const T* block_input = input + block_offset;
    T* block_output = output + ctaid;

    if (ctaid == blockDim.x - 1) {
        int tail = size % BlockReduceFullT::kWorkloadLine;
        BlockReduceTailT block_op;
        block_op(block_input, block_output, tail);
    } else {
        BlockReduceFullT block_op;
        block_op(block_input, block_output, BlockReduceFullT::kWorkloadLine);
    }
}

template<typename T, template<typename> class BinaryOp>
COGITO_KERNEL
void ReduceFinalKernel(const T* input, T* output, int size) {
    using BinaryOpT = BinaryOp<T>;

    BinaryOpT op;
    T res = input[0];

    COGITO_PRAGMA_UNROLL
    for (int i = 1; i < size; ++i){
        res = op(res, input[i]);
    }
    *output = res;
}

} // namsespace detail

///////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, template<typename> class ReduceOp>
struct Reduce {
public:
    static constexpr int kBlockDimX = 256;

public:
    Status operator()(T* input, T* output, int size, cudaStream_t stream = nullptr) {
        dim3 bDim(kBlockDimX);
        dim3 gDim(UPPER_DIV(size, kBlockDimX * 4));

        if (gDim.x == 1) {
            detail::ReduceSingleKernel<T, ReduceOp, kBlockDimX><<<gDim, bDim, 0, stream>>>(input, output, size);
        } else {
            T* global_workspace;
            cudaMallocAsync(&global_workspace, sizeof(T) * gDim.x, stream);

            detail::ReduceSingleKernel<T, ReduceOp, kBlockDimX><<<gDim, bDim, 0, stream>>>(input, global_workspace, size);
            detail::ReduceFinalKernel<T, ReduceOp><<<1, 1>>>(global_workspace, output, gDim.x);

            cudaFreeAsync(global_workspace, stream);
        }
        return Status::kSuccess;
    }
};

} // namespace cogito::general
