//
//
//
//

#pragma once 

namespace cogito {
namespace general {

template<typename T, template<typename> class UnaryOp>
struct ElementWise;

template<typename T, template<typename> class UnaryOp>
struct Reduce;


} // namespace general
} // namespace cogito

#include "cogito/general/elementwise/elementwise.cuh"