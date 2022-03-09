//
//
//
//

#pragma once

#include "cogito/cogito.h"

namespace cogito {
namespace general {
namespace detail {

///////////////////////////////////////////////////////////////////////////////////////////////


template<typename T, template<typename> class ReduceOp, int VecLength>
struct ThreadReduce
{
    static constexpr int kVecLength = VecLength;

    COGITO_DEVICE
    T operator()(T* array){
        T res = array[0];
        ReduceOp<T> op;
        
        COGITO_UNROLL
        for (int i = 1; i < kVecLength; ++i){
            res = op(res, array[i]);
        }
        return res;
    } 
};

} // namespace detail
} // namespace general
} // namespace cogito