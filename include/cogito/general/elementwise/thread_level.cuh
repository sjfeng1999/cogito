//
//
//
//

#pragma once

#include "cogito/cogito.cuh"
#include "cogito/tensor.cuh"

namespace cogito {
namespace general {
namespace detail {

///////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, template<typename> class ElementWiseOp, int ItemPerThread>
struct ThreadElementWise {
public:
    static constexpr int kItemPerThread = ItemPerThread;
    using ShapedTensorT  = ShapedTensor<T, kItemPerThread>;
    using ElementWiseOpT = ElementWiseOp<T>;

public:
    COGITO_DEVICE
    void operator()(const T* input, T* output){
        ElementWiseOpT op;

        COGITO_PRAGMA_UNROLL
        for (int i = 0; i < kItemPerThread; ++i){
            op(input + i, output + i);
        }
    } 

    COGITO_DEVICE
    void operator()(const T* input, T* output, const T& operand){
        ElementWiseOpT op;

        COGITO_PRAGMA_UNROLL
        for (int i = 0; i < kItemPerThread; ++i){
            op(input + i, output + i, operand);
        }
    } 

    COGITO_DEVICE
    void operator()(const ShapedTensorT& input_tensor, ShapedTensorT& output_tensor){
        ElementWiseOpT op;

        COGITO_PRAGMA_UNROLL
        for (int i = 0; i < kItemPerThread; ++i){
            op(&input_tensor[i], &output_tensor[i]);
        }
    } 
};

} // namespace detail
} // namespace general
} // namespace cogito