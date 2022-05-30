//
//
//
//

#pragma once

#include "cogito/cogito.cuh"
#include "cogito/general/reduce/thread_level.cuh"

namespace cogito {
namespace general {
namespace detail {

///////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, template<typename> class ReduceOp, int ItemsPerThread>
struct WarpReduce {
public:
    static constexpr int kItemsPerThread = ItemsPerThread;
    using ReduceOpT     = ReduceOp<T>;
    using ThreadReduceT = ThreadReduce<T, ReduceOp, kItemsPerThread>;
    using ShapedTensorT = ShapedTensor<T, kItemsPerThread>;

public:
    COGITO_DEVICE 
    T operator()(const ShapedTensorT& input){
        ReduceOpT op;
        ThreadReduceT thread_op;
        T val = thread_op(input);

        COGITO_PRAGMA_UNROLL
        for (int offset = 0; offset < 5; ++offset) {
            T shfl_res = __shfl_down_sync(0xffffffff, val, 1 << offset);
            val = op(val, shfl_res);
        }
        return val;
    }

    COGITO_DEVICE 
    T operator()(const ShapedTensorT& input, uint32_t mask){
        ReduceOpT op;
        ThreadReduceT thread_op;
        T val = thread_op(input);

        COGITO_PRAGMA_UNROLL
        for (int offset = 0; offset < 5; ++offset) {
            T shfl_res = __shfl_down_sync(mask, val, 1 << offset);
            val = op(val, shfl_res);
        }
        return val;
    }
};


} // namespace detail
} // namespace general
} // namespace cogito