//
//
//
//

#pragma once

namespace cogito::general::detail {

///////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, template<typename> class ElementwiseOp, int ItemsPerThread>
struct ThreadElementwise {
public:
    static constexpr int kItemsPerThread = ItemsPerThread;
    using ShapedTensorT  = ShapedTensor<T, kItemsPerThread>;
    using ElementwiseOpT = ElementwiseOp<T>;

public:
    template<typename... TensorList>
    COGITO_DEVICE
    void operator()(ShapedTensorT& output_tensor, const TensorList&... input_tensor) {
        ElementwiseOpT op;

        COGITO_PRAGMA_UNROLL
        for (int i = 0; i < kItemsPerThread; ++i) {
            output_tensor[i] = op(input_tensor[i]...);
        }
    } 

    template<typename... TensorList>
    COGITO_DEVICE
    void operator()(ShapedTensorT& output_tensor, const T& operand, const TensorList&... input_tensor) {
        ElementwiseOpT op;

        COGITO_PRAGMA_UNROLL
        for (int i = 0; i < kItemsPerThread; ++i) {
            output_tensor[i] = op(operand, input_tensor[i]...);
        }
    } 
};

} // namespace cogito::general::detail
