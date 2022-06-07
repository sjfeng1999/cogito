//
//
//
//

#pragma once 

namespace cogito {
namespace dnn {

///////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

template<typename T>
COGITO_GLOBAL 
void BatchNorm2dKernel(){
    
}

} // namsespace detail

///////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct BatchNorm2d {


    cudaError_t operator()(T* input, T* output, int n, int c, int h, int w, cudaStream_t stream = nullptr){

    }
};


} // namespace dnn
} // namespace cogito
