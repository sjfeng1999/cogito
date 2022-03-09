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

template <typename T, template<typename> class ReduceOpT>
struct WarpReduce
{   
    COGITO_DEVICE 
    void operator()(T& val){
        for (int i = 0; i < SIZE; ++i){
            val = ReduceOpT<T>()(val, __shfl_down_sync(0xffff, val, mask));
        }
    }
};


} // namespace detail
} // namespace general
} // namespace cogito