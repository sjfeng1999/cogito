//
// 
//
//

#pragma once 

#include <mma.h>

#include "cogito/common/ldst.cuh"
#include "cogito/blas/gemm/thread_level.cuh"

namespace cogito {
namespace blas {
namespace detail {

///////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct FragmentSrcIterator {
public:
    static constexpr int kTileWidth = 128;
    static constexpr int kLoop      = 8;
    static constexpr LoadPolicy kLdPolicy = LoadPolicy::kDefault;

private:
    cogito_shared_ptr T* ptr_;
    cogito_device_reg ShapedTensor<T, 4> frag_[2];
    int strip_offset;
    int block_offset;

public:
    FragmentSrcIterator() = default;

    COGITO_DEVICE
    FragmentSrcIterator(T* shared_ptr) : ptr_(shared_ptr) {
        int tid = threadIdx.x;
        strip_offset = (tid & 0xf) << 2;
        block_offset = (tid >> 4) << 2;
    }

    COGITO_DEVICE
    void reset(T* shared_ptr){ ptr_ = shared_ptr; }

    COGITO_DEVICE
    void stripedLoad(){
        ThreadLd<T, kLdPolicy>::load(frag_[0], ptr_ + strip_offset);
        ThreadLd<T, kLdPolicy>::load(frag_[1], ptr_ + strip_offset + (kTileWidth >> 1));
    }

    COGITO_DEVICE
    void blockedLoad(){
        ThreadLd<T, kLdPolicy>::load(frag_[0], ptr_ + block_offset);
        ThreadLd<T, kLdPolicy>::load(frag_[1], ptr_ + block_offset + (kTileWidth >> 1));
    }

    COGITO_DEVICE
    void operator++(int){ ptr_ += kTileWidth; }

    COGITO_DEVICE 
    constexpr ShapedTensor<T, 4>& operator[](int pos) { return frag_[pos]; }
};


template<typename T>
struct FragmentResIterator {
public:
    static constexpr int kHeightStride = 64;
    static constexpr int kWidthStride  = 64;

private:
    cogito_device_ptr T* ptr_;
    cogito_device_reg ShapedTensor<T, 16> frag_[4];
    const T beta_;
    const int ldg_;

public:
    FragmentResIterator() = default;

    COGITO_DEVICE
    FragmentResIterator(T beta, T* ptr, const int ldg) : beta_(beta), ldg_(ldg) {
        {
            int tid = threadIdx.x;
            ptr_ = ptr + ((tid & 0xf) << 2) + ((tid >> 4) << 2) * ldg_;
        }
        ThreadLd<T>::stripedLoad<0, 4>(frag_[0], ptr_                                      , ldg_, mp::Range2Type<0, 4>{});
        ThreadLd<T>::stripedLoad<0, 4>(frag_[1], ptr_ + kWidthStride                       , ldg_, mp::Range2Type<0, 4>{});
        ThreadLd<T>::stripedLoad<0, 4>(frag_[2], ptr_ + kHeightStride * ldg_               , ldg_, mp::Range2Type<0, 4>{});
        ThreadLd<T>::stripedLoad<0, 4>(frag_[3], ptr_ + kHeightStride * ldg_ + kWidthStride, ldg_, mp::Range2Type<0, 4>{});

        COGITO_PRAGMA_UNROLL
        for (int i = 0; i < 4; ++i) {
            COGITO_PRAGMA_UNROLL
            for (int j = 0; j < 16; ++j) {
                frag_[i][j] *= beta;
            }
        }
    };

    COGITO_DEVICE
    void store() {
        ThreadSt<T>::stripedStore<0, 4>(frag_[0], ptr_                                      , ldg_, mp::Range2Type<0, 4>{});
        ThreadSt<T>::stripedStore<0, 4>(frag_[1], ptr_ + kWidthStride                       , ldg_, mp::Range2Type<0, 4>{});
        ThreadSt<T>::stripedStore<0, 4>(frag_[2], ptr_ + kHeightStride * ldg_               , ldg_, mp::Range2Type<0, 4>{});
        ThreadSt<T>::stripedStore<0, 4>(frag_[3], ptr_ + kHeightStride * ldg_ + kWidthStride, ldg_, mp::Range2Type<0, 4>{});
    }

    COGITO_DEVICE 
    ShapedTensor<T, 16>& operator[](int pos) {
        return frag_[pos];
    }
};


template<typename T, MmaType type = MmaType::kLegacy>
struct WarpMma {
public:
    static constexpr int kM = 16;
    static constexpr int kN = 16;
    static constexpr int kK = 16;
    using ThreadMmaT            = ThreadMma<T>;
    using FragmentSrcAIteratorT = FragmentSrcIterator<T>;
    using FragmentSrcBIteratorT = FragmentSrcIterator<T>;
    using FragmentResIteratorT  = FragmentResIterator<T>;

public:
    COGITO_DEVICE
    void operator()(const T alpha, FragmentSrcAIteratorT& frag_a, FragmentSrcBIteratorT& frag_b, const T beta, FragmentResIteratorT& frag_c) {
        
        ThreadMmaT op;
        for(int i = 0; i < FragmentSrcAIteratorT::kLoop; ++i) {
            frag_a.blockedLoad();
            frag_b.stripedLoad();

            op(alpha, frag_a[0], frag_b[0], beta, frag_c[0]);
            op(alpha, frag_a[0], frag_b[1], beta, frag_c[1]);
            op(alpha, frag_a[1], frag_b[0], beta, frag_c[2]);
            op(alpha, frag_a[1], frag_b[1], beta, frag_c[3]);

            frag_a++;
            frag_b++;
        }
    }
};

/*
template<typename T>
struct WarpMma<T, MmaType::kTensorCore> {
public:
    static constexpr int kM = 16;
    static constexpr int kN = 16;
    static constexpr int kK = 16;
    using FragmentSrcAIteratorT = typename nvcuda::wmma::fragment template<typename nvcuda::wmma::matrix_a, kM, kN, kK, T, typename nvcuda::wmma::row_major>;
    using FragmentSrcBIteratorT = typename nvcuda::wmma::fragment template<typename nvcuda::wmma::matrix_b, kM, kN, kK, T, typename nvcuda::wmma::row_major>;
    using FragmentResIteratorT  = typename nvcuda::wmma::fragment template<typename nvcuda::wmma::accumulator, kM, kN, kK, T>;

public:
    COGITO_DEVICE
    void operator()(T alpha, FragmentSrcAIteratorT frag_a, FragmentSrcBIteratorT frag_b, T beta, FragmentResIteratorT frag_c){
        FragmentResIteratorT frag_acc;
        wmma::mma_sync(acc_frag, frag_a, frag_b, acc_frag);
    }
};

*/

} // namespace detail
} // namespace blas
} // namespace cogito
