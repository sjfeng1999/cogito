//
//
// n-D vector
//

#pragma once 

#include <cstdint>
#include <vector>
#include <utility>
#include <type_traits>

#include "cogito/cogito.cuh"

namespace cogito {

///////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, int... Dims>
struct ShapedTensor {
public:
    static constexpr int kRank        = sizeof...(Dims);
    static constexpr int kDims[]      = {Dims...};
    static constexpr int kSize        = mp::Product<Dims...>::value;
    static constexpr int kElementSize = sizeof(T);
    static_assert(mp::IsPow2<kElementSize>::value);

private:
    cogito_device_reg mutable T data_[kSize];

public:
    ShapedTensor() = default;
    ~ShapedTensor() = default;
    ShapedTensor(const ShapedTensor& tensor) = default;
    ShapedTensor(ShapedTensor&& tensor) = default;

    COGITO_DEVICE 
    ShapedTensor(const T& args...);

    COGITO_DEVICE 
    T& operator[](int pos) { return data_[pos]; }

    COGITO_DEVICE 
    const T& operator[](int pos) const { return data_[pos]; }

    COGITO_DEVICE
    T* data() { return data_; }

    COGITO_DEVICE
    constexpr int size() { return kSize; }

    template<int... Indexes>
    COGITO_DEVICE
    T& at() { 
        static_assert(mp::Product<Indexes...>::value < kSize, "index exceed tensor size");
        return data_[mp::Product<Indexes...>::value]; 
    }

    template<int Start = 0, int Length = kSize>
    COGITO_DEVICE
    void setValue(const T& val) {
        COGITO_PRAGMA_UNROLL
        for (int i = 0; i < kSize; ++i) {
            data_[i] = val;
        }
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, int... Dims>
constexpr ShapedTensor<T, Dims...> make_ShapedTensor(T* data) {
    return ShapedTensor<T, Dims...>(data);
}

template<int dim, typename T, int... Dims>
constexpr int get_dims(ShapedTensor<T, Dims...> /* unuse */) {
    return ShapedTensor<T, Dims...>::kDims[dim];
}

template<int pos, typename T, int... Dims>
constexpr int& get(ShapedTensor<T, Dims...> tensor) {
    return tensor[pos];
}

} // namespace cogito
