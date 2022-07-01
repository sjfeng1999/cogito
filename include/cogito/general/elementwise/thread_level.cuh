//
//
//
//

#pragma once

namespace cogito::general::detail {

///////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, template<typename> class ElementWiseOp, int ItemsPerThread>
struct ThreadElementWise {
public:
    static constexpr int kItemsPerThread = ItemsPerThread;
    using ShapedTensorT  = ShapedTensor<T, kItemsPerThread>;
    using ElementWiseOpT = ElementWiseOp<T>;

public:
    template<typename... TensorList>
    COGITO_DEVICE
    void operator()(ShapedTensorT& output_tensor, const TensorList&... input_tensor) {
        ElementWiseOpT op;

        COGITO_PRAGMA_UNROLL
        for (int i = 0; i < kItemsPerThread; ++i) {
            output_tensor[i] = op(input_tensor[i]...);
        }
    } 

    template<typename... TensorList>
    COGITO_DEVICE
    void operator()(ShapedTensorT& output_tensor, const T& operand, const TensorList&... input_tensor) {
        ElementWiseOpT op;

        COGITO_PRAGMA_UNROLL
        for (int i = 0; i < kItemsPerThread; ++i) {
            output_tensor[i] = op(operand, input_tensor[i]...);
        }
    } 
};

} // namespace cogito::general::detail
