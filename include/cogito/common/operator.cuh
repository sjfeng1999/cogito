//
//
// 
//

#pragma once 

#include "cuda_fp16.h"
#include "cuda_bf16.h"

namespace cogito::op {

///////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct Neg {
    COGITO_DEVICE
    T operator()(const T& a) {
        return -a;
    }
};

template<typename T>
struct Exp {
    COGITO_DEVICE
    T operator()(const T& a) {
        return exp(a);
    }
};

template<typename T>
struct Sum {
    COGITO_DEVICE
    T operator()(const T& a, const T& b) {
        return a + b;
    }
};

template<typename T>
struct Mul {
    COGITO_DEVICE
    T operator()(const T& a, const T& b) {
        return a * b;
    }
};

template<typename T>
struct Div {
    COGITO_DEVICE
    T operator()(const T& a, const T& b) {
        return a / b;
    }
};

template<typename T>
struct Max {
    COGITO_DEVICE
    T operator()(const T& a, const T& b) {
        return max(a, b);
    }
};

template<typename T>
struct Min {
    COGITO_DEVICE
    T operator()(const T& a, const T& b) {
        return min(a, b);
    }
};

} // namespace cogito::op
