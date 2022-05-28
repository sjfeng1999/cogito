//
//
// n-D vector
//

#pragma once 

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
    static constexpr int kSize   = Product<Dims...>::value;
    static constexpr int kElementSize = sizeof(T);

private:
    cogito_device_ptr T data_[kSize];

public:
    ShapedTensor() = default;
    ShapedTensor(T* data) {
        COGITO_PRAGMA_UNROLL
        for (int i = 0; i < kSize; ++i) {
            data_[i] = data[i];
        }
    }

    COGITO_DEVICE 
    T& operator[](int pos) { return data_[pos]; }

    COGITO_DEVICE 
    const T& operator[](int pos) const { return data_[pos]; }

    COGITO_DEVICE
    T* data() { return data_; }

    template<int pos>
    COGITO_DEVICE
    T get() { return data_[pos]; }

    template<int Start = 0, int Length = kSize>
    COGITO_DEVICE
    void load(const T* ptr) {
        COGITO_PRAGMA_UNROLL
        for (int i = Start; i < Start + Length; ++i) {
            data_[i] = *(ptr + i);
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
