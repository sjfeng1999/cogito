//
//
//
//

#pragma once 

namespace cogito::dnn {

///////////////////////////////////////////////////////////////////////////////////////////////

/*****************************************/
// Activation
/*****************************************/

template<typename T>
struct Sigmoid;

enum class SoftmaxType {
    kSharedInternal,
    kDefault,
};

template<typename T, SoftmaxType type>
struct Softmax;

template<typename T>
struct PRelu;

/*****************************************/
// Convolution
/*****************************************/

enum class ConvType {
    kImplicitGemm,
    kWinograd,
};

template<typename T, ConvType type>
struct Convolution2d;


/*****************************************/
// Normalization
/*****************************************/

template<typename T>
struct BatchNorm2d;

template<typename T>
struct LayerNorm2d;

} // namespace cogito::dnn

#include "cogito/dnn/activation/sigmoid.cuh"
#include "cogito/dnn/activation/softmax.cuh"
#include "cogito/dnn/activation/prelu.cuh"

// #include "cogito/dnn/conv/conv.cuh"

// #include "cogito/dnn/normalization/batchnorm2d.cuh"
// #include "cogito/dnn/normalization/layernorm2d.cuh"
