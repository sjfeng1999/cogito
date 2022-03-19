//
//
//
//

#pragma once 

namespace cogito {
namespace dnn {

///////////////////////////////////////////////////////////////////////////////////////////////

// Activation 

template<typename T>
struct Sigmoid;

template<typename T>
struct Softmax;


// Conv 

template<typename T>
struct Convolution2d;


// Norm

template<typename T>
struct BatchNorm2d;

template<typename T>
struct LayerNorm2d;

} // namespace dnn
} // namespace cogito


#include "cogito/dnn/activation/sigmoid.cuh"
#include "cogito/dnn/activation/softmax.cuh"

#include "cogito/dnn/conv/conv2d.cuh"

#include "cogito/dnn/normalization/batchnorm2d.cuh"
#include "cogito/dnn/normalization/layernorm2d.cuh"