//
//
//
//

#pragma once 

#include "cogito/cogito.cuh"
#include "cogito/common/operator.cuh"
#include "cogito/general/general.cuh"

namespace cogito::dnn {

///////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

template<typename T>
struct SubAndExp {
    COGITO_DEVICE
    T operator()(const T& operand, const T& input) {
        return exp(input - operand);
    }
};

template<typename T, SoftmaxType type, int BlockDimX, int ItemsPerThread = 1>
COGITO_KERNEL
void SoftmaxKernel(const int outer_size, const int inner_size, const T* input, T* output) {
    using BlockReduceMaxT = general::detail::BlockReduceAll<T, op::Max, BlockDimX, ItemsPerThread>;
    using BlockSubAndExpT = general::detail::BlockElementwiseAll<T, SubAndExp, BlockDimX, ItemsPerThread>;
    using BlockReduceSumT = general::detail::BlockReduceAll<T, op::Sum, BlockDimX, ItemsPerThread>;
    using BlockDivT       = general::detail::BlockElementwiseAll<T, op::Div, BlockDimX, ItemsPerThread>;

    __shared__ T internal_val;

    const int ctaid = blockIdx.x;
    const int block_offset = ctaid * inner_size;

    const T* block_input = input + block_offset;
    T* block_output = output + block_offset;

    BlockReduceMaxT block_reduce_max_op;
    BlockSubAndExpT block_sub_exp_op;
    BlockReduceSumT block_reduce_sum_op;
    BlockDivT block_div_op;

    T* internal_workspace;
    if constexpr (type == SoftmaxType::kSharedInternal) {
        extern __shared__  T shared_workspace[];
        internal_workspace = shared_workspace;
    } else {
        internal_workspace = block_output;
    }

    block_reduce_max_op(block_input, &internal_val, inner_size);
    block_sub_exp_op(block_input, internal_workspace, internal_val, inner_size);
    __syncthreads();
    block_reduce_sum_op(internal_workspace, &internal_val, inner_size);
    block_div_op(internal_workspace, block_output, internal_val, inner_size);
}

} // namsespace detail

///////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, SoftmaxType type = SoftmaxType::kDefault>
struct Softmax {
public:
    static constexpr int kBlockDimX = 256;
    static constexpr SoftmaxType kType = type;

public:
    Status operator()(const int batch_size, const int inner_size, const T* input, T* output, cudaStream_t stream = nullptr) {
        dim3 bDim(kBlockDimX);
        dim3 gDim(batch_size);
        size_t shared_workspace = 0;
        
        if constexpr (kType == SoftmaxType::kSharedInternal) {
            shared_workspace = inner_size * sizeof(T);
        }

        if (inner_size % 4 == 0) {
            detail::SoftmaxKernel<T, kType, kBlockDimX, 4><<<gDim, bDim, shared_workspace, stream>>>(batch_size, inner_size, input, output);
        } else if (inner_size % 2 == 0) {
            detail::SoftmaxKernel<T, kType, kBlockDimX, 2><<<gDim, bDim, shared_workspace, stream>>>(batch_size, inner_size, input, output);
        } else {
            detail::SoftmaxKernel<T, kType, kBlockDimX, 1><<<gDim, bDim, shared_workspace, stream>>>(batch_size, inner_size, input, output);
        }
        return Status::kSuccess;
    }
};

} // namespace cogito::dnn
