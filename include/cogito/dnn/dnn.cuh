//
//
//
//

#pragma once 

namespace cogito {
namespace dnn {

///////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct Sigmoid;

template<typename T>
struct Softmax;

template<typename T>
struct PRelu;

enum class ConvType {
    kImplicitGemm,
    kWinograd,
};

template<typename T, ConvType type>
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
#include "cogito/dnn/activation/prelu.cuh"

// #include "cogito/dnn/conv/conv.cuh"

// #include "cogito/dnn/normalization/batchnorm2d.cuh"
// #include "cogito/dnn/normalization/layernorm2d.cuh"
