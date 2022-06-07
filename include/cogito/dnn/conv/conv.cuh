//
//
//
//

#pragma once 

#include "cogito/cogito.cuh"
#include "cogito/dnn/dnn.cuh"

namespace cogito {
namespace dnn {

///////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

template<typename T>
COGITO_KERNEL
void im2col(T* input, 
            int n, int c, int h, int w, int q, int r, int s, int k, 
            cudaStream_t stream = nullptr){
    int tid = threadIdx.x;

}

template<typename T>
COGITO_KERNEL
void col2img(T* input, 
            int n, int c, int h, int w, int q, int r, int s, int k, 
            cudaStream_t stream = nullptr){
    int tid = threadIdx.x;

}

template<typename T>
COGITO_GLOBAL 
void Conv2dKernel(T* input, T* output, T* kernel, 
                  int n, int c, int h, int w, int q, int r, int s, int k, 
                  cudaStream_t stream = nullptr) {
    
    
    int tid = threadIdx.x;

}

} // namsespace detail

///////////////////////////////////////////////////////////////////////////////////////////////


template<typename T, ConvType type>
struct Convolution2d {

    Status operator()(T* input, T* output, T* kernel, 
                           int n, int c, int h, int w, int q, int r, int s, int k, 
                           cudaStream_t stream = nullptr) {
        Status status = Status::kOK; 

        int x = c;

        return status;
    }
};

} // namespace dnn
} // namespace cogito
