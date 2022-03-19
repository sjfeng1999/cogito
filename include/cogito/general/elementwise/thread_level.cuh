//
//
//
//

#pragma once

#include "cogito/cogito.cuh"

namespace cogito {
namespace general {
namespace detail {

///////////////////////////////////////////////////////////////////////////////////////////////


template<typename T, template<typename> class ElementWiseOp, int VecLength = 1>
struct ThreadElementWise
{
    static constexpr int kVecLength = VecLength;
    
    using ElementWiseOpT = ElementWiseOp<T>;


    COGITO_DEVICE
    void operator()(T* input, T* output){
        ElementWiseT op;

        COGITO_UNROLL
        for (int i = 0; i < kVecLength; ++i){
            output[i] = op(input[i]);
        }
    } 

    COGITO_DEVICE
    void operator()(T* input, T* output, const T& operand){
        ElementWiseT op;

        COGITO_UNROLL
        for (int i = 0; i < kVecLength; ++i){
            output[i] = op(input[i], operand);
        }
    } 
};

} // namespace detail
} // namespace general
} // namespace cogito