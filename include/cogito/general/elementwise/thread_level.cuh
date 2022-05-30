//
//
//
//

#pragma once

namespace cogito {
namespace general {
namespace detail {

///////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, template<typename> class ElementWiseOp, int ItemsPerThread>
struct ThreadElementWise {
public:
    static constexpr int kItemsPerThread = ItemsPerThread;
    using ElementWiseOpT = ElementWiseOp<T>;
    using ShapedTensorT  = ShapedTensor<T, kItemsPerThread>;

public:
    COGITO_DEVICE
    void operator()(const ShapedTensorT& input_tensor, ShapedTensorT& output_tensor){
        ElementWiseOpT op;

        COGITO_PRAGMA_UNROLL
        for (int i = 0; i < kItemsPerThread; ++i){
            output_tensor[i] = op(input_tensor[i]);
        }
    } 

    COGITO_DEVICE
    void operator()(const ShapedTensorT& input_tensor, ShapedTensorT& output_tensor, const T& operand){
        ElementWiseOpT op;

        COGITO_PRAGMA_UNROLL
        for (int i = 0; i < kItemsPerThread; ++i){
            output_tensor[i] = op(input_tensor[i], operand);
        }
    } 
};

} // namespace detail
} // namespace general
} // namespace cogito
