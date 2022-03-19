//
//
//
//

#pragma once 

#include "cogito/cogito.cuh"

#include "cogito/general/general.cuh"

namespace cogito {
namespace dnn {

///////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

template<typename T>
struct Sigmoid {
    COGITO_HOST_DEVICE
    void operator()(T* input, T* output){
        T val = *input;
        *output = 1 / (1 + exp(-val));
    }
};

} // namsespace detail

///////////////////////////////////////////////////////////////////////////////////////////////


template<typename T>
struct Sigmoid 
{
    using ElementWiseKernelT = general::ElementWise<T, detail::Sigmoid>;

    cudaError_t operator()(T* input, T* output, int size){
        ElementWiseKernelT op;
        return op(input, output, size);
    }
};


} // namespace dnn
} // namespace cogito