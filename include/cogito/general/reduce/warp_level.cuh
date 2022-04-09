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

template <typename T, template<typename> class ReduceOp>
struct WarpReduce {
          
    using ReduceOpT = ReduceOp<T>;

    COGITO_DEVICE 
    T operator()(T* input){
        ReduceOpT op;
        T val = input[0];

        COGITO_PRAGMA_UNROLL
        for (int offset = 0; offset < 5; ++offset) {
            T shfl_res = __shfl_down_sync(0xffffffff, val, 1 << offset);
            val = op(&val, &shfl_res);
        }
        return val;
    }
};


} // namespace detail
} // namespace general
} // namespace cogito