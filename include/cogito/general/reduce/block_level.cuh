//
//
//
//

#pragma once

#include "cogito/ptx.cuh"
#include "cogito/general/reduce/warp_level.cuh"

namespace cogito::general::detail {

///////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, template<typename> class ReduceOp, int BlockDimX, int blockSize, int stripSize, bool Full>
struct BlockReduce {
public:
    static constexpr bool kFull        = Full;
    static constexpr int kBlockDimX    = BlockDimX;
    static constexpr int kBlockSize    = blockSize;
    static constexpr int kStripSize    = stripSize;
    static constexpr int kWarpNums     = kBlockDimX / kWarpSize;
    static constexpr int kWorkloadLine = kBlockDimX * kBlockSize * kStripSize;
    static constexpr LoadPolicy  kLdPolicy = LoadPolicy::kCS;
    static constexpr StorePolicy kStPolicy = StorePolicy::kWT;
    using ShapedTensorT = ShapedTensor<T, kBlockSize>;
    using ReduceOpT     = ReduceOp<T>;
    using WarpReduceT   = WarpReduce<T, ReduceOp, kBlockSize>;
    using ThreadReduceElementwiseOpT = ThreadElementwise<T, ReduceOp, kBlockSize>;

public:
    COGITO_DEVICE 
    void operator()(const T* input, T* output, const int size) {
        int tid = threadIdx.x;
        int offset = tid * kBlockSize;

        ShapedTensorT input_tensor;

        if constexpr (kFull) {
            ThreadLd<T>::load(input_tensor, input + offset);
            if constexpr (kStripSize > 1) {
                ShapedTensorT tmp_tensor;
                ThreadReduceElementwiseOpT element_reduce_op;

                COGITO_PRAGMA_UNROLL
                for (int i = 1; i < kStripSize; ++i) {
                    offset += kWorkloadLine;
                    ThreadLd<T>::load(tmp_tensor, input + offset);
                    element_reduce_op(input_tensor, input_tensor, tmp_tensor);
                }
            }
        } else {
            const T idendity = ReduceOpT::kIdentity;
            if (offset < size) {
                ThreadLd<T>::load(input_tensor, input + offset);
            } else {
                ThreadLd<T>::load(input_tensor, idendity);
            }
            ShapedTensorT tmp_tensor;
            ThreadReduceElementwiseOpT element_reduce_op;

            offset += kWorkloadLine;
            for (; offset < size; offset += kWorkloadLine) {
                ThreadLd<T>::load(tmp_tensor, input + offset);
                element_reduce_op(input_tensor, input_tensor, tmp_tensor);
            }
        }

        T warp_res;
        {
            WarpReduceT warp_op;
            warp_res = warp_op(input_tensor);
        }

        int laneid = ptx::getLaneid();
        int warpid = tid >> 5;

        __shared__ T warp_aggregates[kWarpNums];

        if (laneid == 0) {
            warp_aggregates[warpid] = warp_res;
        }
        __syncthreads();

        if (tid == 0) {
            ReduceOpT reduce_op;
            COGITO_PRAGMA_UNROLL
            for (int i = 1; i < kWarpNums; ++i){
                warp_res = reduce_op(warp_res, warp_aggregates[i]);
            }
            *output = warp_res;
        }
    }
};


template <typename T, template<typename> class ReduceOp, int BlockDimX, int blockSize>
struct BlockReduceAll {
public:
    static constexpr int kBlockDimX    = BlockDimX;
    static constexpr int kBlockSize    = blockSize;
    static constexpr int kWarpNums     = kBlockDimX / kWarpSize;
    static constexpr int kWorkloadLine = kBlockDimX * kBlockSize;
    static constexpr LoadPolicy  kLdPolicy = LoadPolicy::kCS;
    static constexpr StorePolicy kStPolicy = StorePolicy::kWT;
    using ShapedTensorT = ShapedTensor<T, kBlockSize>;
    using ReduceOpT     = ReduceOp<T>;
    using WarpReduceT   = WarpReduce<T, ReduceOp, kBlockSize>;
    using ThreadReduceElementwiseOpT = ThreadElementwise<T, ReduceOp, kBlockSize>;

public:
    COGITO_DEVICE 
    void operator()(const T* input, T* output, int size) {
        int tid = threadIdx.x;
        int offset = tid * kBlockSize;

        ShapedTensorT input_tensor, tmp_tensor;
        ThreadReduceElementwiseOpT element_reduce_op;
        ReduceOpT reduce_op;

        const T idendity = ReduceOpT::kIdentity;
        if (offset < size) {
            ThreadLd<T>::load(input_tensor, input + offset);
        } else {
            ThreadLd<T>::load(input_tensor, idendity);
        }

        offset += kWorkloadLine;
        for (; offset < size; offset += kWorkloadLine) {
            ThreadLd<T>::load(tmp_tensor, input + offset);
            element_reduce_op(input_tensor, input_tensor, tmp_tensor);
        }

        T warp_res;
        {
            WarpReduceT warp_op;
            warp_res = warp_op(input_tensor);
        }

        int laneid = ptx::getLaneid();
        int warpid = tid >> 5;

        __shared__ T warp_aggregates[kWarpNums];

        if (laneid == 0){
            warp_aggregates[warpid] = warp_res;
        }
        __syncthreads();

        if (tid == 0) {
            COGITO_PRAGMA_UNROLL
            for (int i = 1; i < kWarpNums; ++i) {
                warp_res = reduce_op(warp_res, warp_aggregates[i]);
            }
            *output = warp_res;
        }
    }

};

} // namespace cogito::general::detail
