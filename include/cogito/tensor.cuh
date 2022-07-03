//
// N-d vector with static shape
// 
//

#pragma once 

#include <cstdint>
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
    using type = T;

private:
    cogito_device_reg mutable T data_[kSize];

public:
    ShapedTensor() = default;
    ~ShapedTensor() = default;
    ShapedTensor(const ShapedTensor& tensor) = default;
    ShapedTensor(ShapedTensor&& tensor) = delete;
    ShapedTensor& operator=(const ShapedTensor& tensor) = default;
    ShapedTensor& operator=(ShapedTensor&& tensor) = delete;

    COGITO_DEVICE 
    ShapedTensor(const T& val) {
        COGITO_PRAGMA_UNROLL
        for (int i = 0; i < kSize; ++i) {
            data_[i] = val;
        }
    }

    COGITO_DEVICE 
    ShapedTensor(const T (&input)[kSize]) {
        COGITO_PRAGMA_UNROLL
        for (int i = 0; i < kSize; ++i) {
            data_[i] = input[i];
        }
    }

    COGITO_DEVICE 
    T& operator[](const int pos) { return data_[pos]; }

    COGITO_DEVICE 
    const T& operator[](const int pos) const { return data_[pos]; }

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
        for (int i = 0; i < Length; ++i) {
            data_[i + Start] = val;
        }
    }
};

} // namespace cogito
