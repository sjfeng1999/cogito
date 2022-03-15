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


template<typename T, template<typename> class ReduceOp, int VecLength = 1>
struct ThreadReduce
{
    static constexpr int kVecLength = VecLength;

    COGITO_DEVICE
    T operator()(T* input){
        T res = input[0];
        ReduceOp<T> op;
        
        COGITO_UNROLL
        for (int i = 1; i < kVecLength; ++i){
            res = op(res, input[i]);
        }
        return res;
    } 
};

} // namespace detail
} // namespace general
} // namespace cogito