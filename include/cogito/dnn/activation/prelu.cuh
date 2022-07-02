//
//
//
//

#pragma once 

#include "cogito/cogito.cuh"
#include "cogito/general/general.cuh"

namespace cogito::dnn {

///////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

template<typename T>
struct PReluOp {
    COGITO_DEVICE
    T operator()(const T& input, const T& alpha) {
        return input > 0 ? input : alpha * input;
    }
};

template<typename T, int BlockDimX, int ItemPerThread = 1>
COGITO_KERNEL
void PReluKernel(const int outer_size, const int inner_size, const T* input, const T* alpha, T* output) {
    using BlockElementWiseT = general::detail::BlockElementWiseAll<T, PReluOp, BlockDimX, ItemPerThread>;

    int ctaid = blockIdx.x;
    int block_offset = ctaid * inner_size;

    const T* block_alpha = alpha + ctaid;
    const T* block_input = input + block_offset;
    T* block_output = output + block_offset;

    BlockElementWiseT block_op;
    block_op(block_input, block_output, block_alpha, inner_size);
}

} // namsespace detail

///////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct PRelu {
public:
    static constexpr int kBlockDimX = 256;

public:
    cudaError_t operator()(const int batch_size, const int inner_size, const T* input, const T* alpha, T* output, cudaStream_t stream = nullptr) {
        
        dim3 bDim(kBlockDimX);
        dim3 gDim(batch_size);

        if (inner_size % 4 == 0 or inner_size % 2 == 0) {
            detail::PReluKernel<T, kBlockDimX, 4><<<gDim, bDim, 0, stream>>>(batch_size, inner_size, input, alpha, output);
        } else if (inner_size % 2 == 0) {
            detail::PReluKernel<T, kBlockDimX, 2><<<gDim, bDim, 0, stream>>>(batch_size, inner_size, input, alpha, output);
        } else {
            detail::PReluKernel<T, kBlockDimX, 1><<<gDim, bDim, 0, stream>>>(batch_size, inner_size, input, alpha, output);
        }

        return general::ElementWise<T, detail::PReluOp>()(batch_size, inner_size, input, alpha, output, stream);
    }
};

} // namespace cogito::dnn
