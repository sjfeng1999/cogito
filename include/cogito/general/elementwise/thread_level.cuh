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
struct ThreadElementWise {

    static constexpr int kVecLength = VecLength;
    
    using ElementWiseOpT = ElementWiseOp<T>;

    COGITO_DEVICE
    void operator()(T* input, T* output){
        ElementWiseOpT op;

        COGITO_PRAGMA_UNROLL
        for (int i = 0; i < kVecLength; ++i){
            op(input + i, output + i);
        }
    } 

    COGITO_DEVICE
    void operator()(T* input, T* output, const T& operand){
        ElementWiseOpT op;

        COGITO_PRAGMA_UNROLL
        for (int i = 0; i < kVecLength; ++i){
            op(input + i, output + i, operand);
        }
    } 
};

} // namespace detail
} // namespace general
} // namespace cogito