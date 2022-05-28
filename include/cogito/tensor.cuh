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
    static constexpr int kRank   = sizeof...(Dims);
    static constexpr int kDims[] = {Dims...};
    static constexpr int kSize   = mp::Product<Dims...>::value;
    static constexpr int kElementSize = sizeof(T);
    static_assert(mp::IsPow2<kElementSize>::value);

private:
    cogito_device_reg mutable T data_[kSize];

public:
    ShapedTensor() = default;

    COGITO_DEVICE 
    T& operator[](int pos) { return data_[pos]; }

    COGITO_DEVICE 
    const T& operator[](int pos) const { return data_[pos]; }

    COGITO_DEVICE
    T* data() { return data_; }

    template<int pos>
    COGITO_DEVICE
    T get() { return data_[pos]; }

    // Load by LDG/S.128
    template<int Start = 0, int Length = kSize, 
        typename std::enable_if<Length * kElementSize % 16 == 0, int>::type Factor = (Length * kElementSize >> 4)>
    COGITO_DEVICE
    void load(const T* ptr) {
        COGITO_PRAGMA_UNROLL
        for (int i = 0; i < Factor; ++i) {
            float4 val = *reinterpret_cast<const float4*>(reinterpret_cast<const int8_t*>(ptr) + i * 32);
            data_[Start + i * 4 + 0] = reinterpret_cast<T&>(val.x);
            data_[Start + i * 4 + 1] = reinterpret_cast<T&>(val.y);
            data_[Start + i * 4 + 2] = reinterpret_cast<T&>(val.z);
            data_[Start + i * 4 + 3] = reinterpret_cast<T&>(val.w);
        }
    }

    // Load by LDG/S.64
    template<int Start = 0, int Length = kSize, 
        typename std::enable_if<(Length * kElementSize % 16 != 0) && (Length * kElementSize % 8 == 0), int>::type Factor = (Length * kElementSize >> 3)>
    COGITO_DEVICE
    void load(const T* ptr) {
        COGITO_PRAGMA_UNROLL
        for (int i = 0; i < Factor; ++i) {
            float2 val = *reinterpret_cast<const float2*>(reinterpret_cast<const int8_t*>(ptr) + i * 16);
            data_[Start + i * 2 + 0] = reinterpret_cast<T&>(val.x);
            data_[Start + i * 2 + 1] = reinterpret_cast<T&>(val.y);
        }
    }

    template<int Start = 0, int Length = kSize, 
        typename std::enable_if<(Length * kElementSize % 16 != 0) && (Length * kElementSize % 8 != 0), int>::type = 0>
    COGITO_DEVICE
    void load(const T* ptr) {
        COGITO_PRAGMA_UNROLL
        for (int i = Start; i < Start + Length; ++i) {
            data_[i] = ptr[i];
        }
    }


    // Store by LDG/S.128
    template<int Start = 0, int Length = kSize, 
        typename std::enable_if<Length * kElementSize % 16 == 0, int>::type Factor = (Length * kElementSize >> 4)>
    COGITO_DEVICE
    void store(T* ptr) const {
        COGITO_PRAGMA_UNROLL
        for (int i = 0; i < Factor; ++i) {
            float4 val;
            val.x = reinterpret_cast<T&>(data_[Start + i * 4 + 0]);
            val.y = reinterpret_cast<T&>(data_[Start + i * 4 + 1]);
            val.z = reinterpret_cast<T&>(data_[Start + i * 4 + 2]);
            val.w = reinterpret_cast<T&>(data_[Start + i * 4 + 3]);
            *reinterpret_cast<float4*>(reinterpret_cast<int8_t*>(ptr) + i * 32) = val;
        }
    }

    // Store by LDG/S.64
    template<int Start = 0, int Length = kSize, 
        typename std::enable_if<(Length * kElementSize % 16 != 0) && (Length * kElementSize % 8 == 0), int>::type Factor = (Length * kElementSize >> 3)>
    COGITO_DEVICE
    void store(T* ptr) const {
        COGITO_PRAGMA_UNROLL
        for (int i = 0; i < Factor; ++i) {
            float2 val;
            val.x = reinterpret_cast<T&>(data_[Start + i * 2 + 0]);
            val.y = reinterpret_cast<T&>(data_[Start + i * 2 + 1]);
            *reinterpret_cast<float2*>(reinterpret_cast<int8_t*>(ptr) + i * 16) = val;
        }
    }

    template<int Start = 0, int Length = kSize, 
        typename std::enable_if<(Length * kElementSize % 16 != 0) && (Length * kElementSize % 8 != 0), int>::type = 0>
    COGITO_DEVICE
    void store(T* ptr) const {
        COGITO_PRAGMA_UNROLL
        for (int i = Start; i < Start + Length; ++i) {
            ptr[i] = data_[i];
        }
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct Tensor {
public:
    static constexpr int kElementSize = sizeof(T);

private:
    int rank_;
    int* dims_;
    int size_;
    cogito_device_ptr T* data_;

public:
    Tensor() = default;
    Tensor(T* data, int rank, int* dims) : data_(data), rank_(rank), dims_(dims), size_(1){
        for (int i = 0; i < rank_; ++i){
            size_ *= dims[i];
        }
    }

    template<typename Element, int... Dims>
    Tensor(const ShapedTensor<Element, Dims...> shaped_tensor) : data_(shaped_tensor.data()), 
                                                                 rank_(ShapedTensor<Element, Dims...>::kRank),
                                                                 // dims_(ShapedTensor<Element, Dims...>::kDims),
                                                                 size_(ShapedTensor<Element, Dims...>::kSize) {}

    T* data() { return data_; }
    int size() { return size_; }
};

///////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
constexpr Tensor<T> make_Tensor(T* data, int rank, int* dims) {
    return Tensor<T>(data, rank, dims);
}

template<typename T, int... Dims>
constexpr ShapedTensor<T, Dims...> make_ShapedTensor(T* data) {
    return ShapedTensor<T, Dims...>(data);
}

template<int dim, typename T, int... Dims>
constexpr int get_dims(ShapedTensor<T, Dims...> tensor) {
    return ShapedTensor<T, Dims...>::kDims[dim];
}

template<int pos, typename T, int... Dims>
constexpr int& get(ShapedTensor<T, Dims...> tensor) {
    return tensor.data()[pos];
}

} // namespace cogito
