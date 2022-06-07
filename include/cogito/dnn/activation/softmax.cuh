//
//
//
//

#pragma once 

#include <limits>
#include "cogito/cogito.cuh"

#include "cogito/general/general.cuh"

namespace cogito {
namespace dnn {

///////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

// ReduceOp
template<typename T>
struct Max {
    static constexpr T kIdentity = std::numeric_limits<T>::min();
    COGITO_DEVICE
    T operator()(const T& left, const T& right) {
        return max(left, right);
    }
};

// ReduceOp
template<typename T>
struct Sum {
    static constexpr T kIdentity = 0;
    COGITO_DEVICE
    T operator()(const T& left, const T& right) {
        return left + right;
    }
};

// ElementWiseOp
template<typename T>
struct Div {
    COGITO_DEVICE
    T operator()(const T& input, const T& operand) {
        return input / operand;
    }
};

// ElementWiseOp
template<typename T>
struct SubAndExp {
    COGITO_DEVICE
    T operator()(const T& input, const T& operand) {
        return exp(input - operand);
    }
};


} // namsespace detail

///////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct Softmax 
{
    using ReduceMaxT         = general::Reduce<T, detail::Max>;
    using ReduceSumT         = general::Reduce<T, detail::Sum>;
    using ElementWiseSubExpT = general::ElementWise<T, detail::SubAndExp>;
    using ElementWiseDivT    = general::ElementWise<T, detail::Div>;

    cudaError_t operator()(T* input, T* output, int size){

        ReduceMaxT()(input, output, size);
        ElementWiseSubExpT()(input, input, output, size);

        ReduceSumT()(input, output, size);
        ElementWiseDivT()(input, output, output, size);

        return cudaPeekAtLastError();
    }
};

} // namespace dnn
} // namespace cogito
