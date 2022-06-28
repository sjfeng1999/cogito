//
//
//
//

#pragma once

namespace cogito {
namespace general {
namespace detail {

///////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, template<typename> class ReduceOp, int ItemsPerThread>
struct ThreadReduce {
public:
    static constexpr int kItemsPerThread = ItemsPerThread;
    using ReduceOpT     = ReduceOp<T>;
    using ShapedTensorT = ShapedTensor<T, kItemsPerThread>;

public:
    COGITO_DEVICE
    T operator()(const ShapedTensorT& input_tensor) {
        ReduceOpT op;
        T res = input_tensor[0];
        
        COGITO_PRAGMA_UNROLL
        for (int i = 1; i < kItemsPerThread; ++i) {
            res = op(res, input_tensor[i]);
        }
        return res;
    } 
};

} // namespace detail
} // namespace general
} // namespace cogito
