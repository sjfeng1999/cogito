//
//
//
//

#pragma once 

namespace cogito::general {

///////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, template<typename> class UnaryOp>
struct ElementWise;

template<typename T, template<typename> class BinaryOp>
struct Reduce;

template<typename T, template<typename> class BinaryOp>
struct Scan;

} // namespace cogito::general

#include "cogito/general/elementwise/elementwise.cuh"
#include "cogito/general/reduce/reduce.cuh"
