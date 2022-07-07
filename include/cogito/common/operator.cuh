//
//
// 
//

#pragma once 

#include <limits>
#include "cuda_fp16.h"
#include "cuda_bf16.h"

namespace cogito::op {

///////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct Neg {
public:
    COGITO_DEVICE
    T operator()(const T& input) const {
        return -input;
    }
};

template<typename T>
struct Exp {
public:
    COGITO_DEVICE
    T operator()(const T& input) const {
        return exp(input);
    }
};

template<typename T>
struct Sum {
public:
    static constexpr T kIdentity = static_cast<T>(0);
    COGITO_DEVICE
    T operator()(const T& a, const T& b) const {
        return a + b;
    }
};

template<typename T>
struct Mul {
public:
    static constexpr T kIdentity = static_cast<T>(1);
    COGITO_DEVICE
    T operator()(const T& a, const T& b) const {
        return a * b;
    }
};

template<typename T>
struct Div {
public:
    COGITO_DEVICE
    T operator()(const T& operand, const T& input) const {
        return input / operand;
    }
};

template<typename T>
struct Max {
public:
    static constexpr T kIdentity = std::numeric_limits<T>::min();
    COGITO_DEVICE
    T operator()(const T& a, const T& b) const {
        return max(a, b);
    }
};

template<typename T>
struct Min {
public:
    static constexpr T kIdentity = std::numeric_limits<T>::max();
    COGITO_DEVICE
    T operator()(const T& a, const T& b) const {
        return min(a, b);
    }
};

} // namespace cogito::op
