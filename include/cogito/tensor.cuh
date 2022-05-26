//
//
// n-D vector
//

#pragma once 

#include <vector>
#include <utility>
#include "cogito/cogito.cuh"

namespace cogito {

///////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, int Rank, int... Dims>
class ShapedTensor {

public:
    static_assert(Rank == sizeof...(Dims));
    static constexpr int kRank = Rank;

private:
    T* data_;
    const int dims_[kRank] = {0};
    int size_;

public:
    ShapedTensor() = default;

    T* data();
    int size();
};

///////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, int Rank>
class RankedTensor {

public:
    static constexpr int kRank = Rank;

private:
    T* data_;
    int dims_[kRank];
    int size_;

public:
    RankedTensor();


    T* data() { return data_; }
    int size() { return size_; }
};

///////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
class Tensor {
public:
    using value_type = T;
    static constexpr int kElementSize = sizeof(T);
    
private:
    cogito_device_ptr T* data_;
    int rank_;
    int* dims_;
    int size_;

public:
    Tensor() = default;
    Tensor(T* data, int rank, int* dims) : data_(data), rank_(rank), dims_(dims), size_(1){
        for (int i = 0; i < rank_; ++i){
            size_ *= dims[i];
        }
    }
    template<typename Element, int Rank>
    Tensor(const RankedTensor<Element, Rank> ranked_tensor);

    template<typename Element, int Rank, int... Dims>
    Tensor(const ShapedTensor<Element, Rank, Dims...> shaped_tensor);

    T* data() { return data_; }
    int size() { return size_; }
};

///////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
Tensor<T> make_Tensor(T* data, int rank, int* dims) {
    Tensor<T> tensor(data, rank, dims);
    return tensor;
}



} // namespace cogito
