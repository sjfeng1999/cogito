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

template<int M, int N, int K>
struct GemmShape;

template<typename T, bool mat_a>
struct FragmentSrcIterator {
public:
    static constexpr int kTileWidth = mat_a ? 140 : 128;
    static constexpr int kLoop      = 8;
    static constexpr LoadPolicy kLdPolicy = LoadPolicy::kDefault;

private:
    cogito_shared_ptr T* ptr_;
    cogito_device_reg ShapedTensor<T, 4> frag_[2];
    int offset_;

public:
    FragmentSrcIterator() = delete;

    COGITO_DEVICE
    FragmentSrcIterator(T* shared_ptr) : ptr_(shared_ptr) {
        int tid = threadIdx.x;
        if constexpr (mat_a) {
            offset_ = (tid >> 4) << 2;
        } else {
            offset_ = (tid & 0xf) << 2;
        }
    }

    COGITO_DEVICE
    void reset(T* shared_ptr){ ptr_ = shared_ptr; }

    COGITO_DEVICE
    void load() {
        if constexpr (mat_a) {
            ThreadLd<T, kLdPolicy>::load(frag_[0], ptr_ + offset_);
            ThreadLd<T, kLdPolicy>::load(frag_[1], ptr_ + offset_ + 72);
        } else {
            ThreadLd<T, kLdPolicy>::load(frag_[0], ptr_ + offset_);
            ThreadLd<T, kLdPolicy>::load(frag_[1], ptr_ + offset_ + 64);
        }
    }

    COGITO_DEVICE
    void operator++(int) { ptr_ += kTileWidth; }

    COGITO_DEVICE 
    constexpr ShapedTensor<T, 4>& operator[](int pos) { return frag_[pos]; }
};


template<typename T>
struct FragmentResIterator {
public:
    static constexpr int kHeightStride = 64;
    static constexpr int kWidthStride  = 64;
    static constexpr LoadPolicy  kLdPolicy = LoadPolicy::kCA;
    static constexpr StorePolicy kStPolicy = StorePolicy::kDefault;

private:
    cogito_device_ptr T* ptr_;
    cogito_device_ptr T* dest_ptr_[4];
    cogito_device_reg ShapedTensor<T, 16> frag_[4];
    cogito_device_reg ShapedTensor<T, 16> store_;
    int offset_;
    const T beta_;
    const int ldg_;

public:
    FragmentResIterator() = default;

    COGITO_DEVICE
    FragmentResIterator(T beta, T* ptr, const int ldg) : beta_(beta), ptr_(ptr), ldg_(ldg) {
        int tid = threadIdx.x;
        offset_ = ((tid & 0xf) << 2) + ((tid >> 4) << 2) * ldg_;

        COGITO_PRAGMA_UNROLL
        for (int i = 0; i < 4; ++i) {
            frag_[i].setValue(0);
        }
        dest_ptr_[0] = ptr_ + offset_;
        dest_ptr_[1] = ptr_ + offset_ + kWidthStride;
        dest_ptr_[2] = ptr_ + offset_ + kHeightStride * ldg_;
        dest_ptr_[3] = ptr_ + offset_ + kHeightStride * ldg_ + kWidthStride;
        ThreadLd<T, kLdPolicy>::stripedLoad<0, 4>(store_, dest_ptr_[0], ldg_, mp::Range2Type<0, 4>{});
        // ThreadLd<T, kLdPolicy>::stripedLoad<0, 4>(store_, dest_ptr_[1], ldg_, mp::Range2Type<0, 4>{});
    };

    COGITO_DEVICE
    void store() {

        COGITO_PRAGMA_UNROLL
        for (int i = 0; i < 16; ++i) {
            store_[i] = beta_ * store_[i] + frag_[0][i];
        }
        ThreadSt<T, kStPolicy>::stripedStore<0, 4>(store_, dest_ptr_[0], ldg_, mp::Range2Type<0, 4>{});
        ThreadLd<T, kLdPolicy>::stripedLoad<0, 4>(store_, dest_ptr_[1], ldg_, mp::Range2Type<0, 4>{});

        COGITO_PRAGMA_UNROLL
        for (int i = 0; i < 16; ++i) {
            store_[i] = beta_ * store_[i] + frag_[1][i];
        }

        ThreadSt<T, kStPolicy>::stripedStore<0, 4>(store_, dest_ptr_[1], ldg_, mp::Range2Type<0, 4>{});
        ThreadLd<T, kLdPolicy>::stripedLoad<0, 4>(store_, dest_ptr_[2], ldg_, mp::Range2Type<0, 4>{});

        COGITO_PRAGMA_UNROLL
        for (int i = 0; i < 16; ++i) {
            store_[i] = beta_ * store_[i] + frag_[2][i];
        }
        ThreadSt<T, kStPolicy>::stripedStore<0, 4>(store_, dest_ptr_[2], ldg_, mp::Range2Type<0, 4>{});
        ThreadLd<T, kLdPolicy>::stripedLoad<0, 4>(store_, dest_ptr_[3], ldg_, mp::Range2Type<0, 4>{});

        COGITO_PRAGMA_UNROLL
        for (int i = 0; i < 16; ++i) {
            store_[i] = beta_ * store_[i] + frag_[3][i];
        }
        ThreadSt<T, kStPolicy>::stripedStore<0, 4>(store_, dest_ptr_[3], ldg_, mp::Range2Type<0, 4>{});
    }

    COGITO_DEVICE 
    ShapedTensor<T, 16>& operator[](int pos) {
        return frag_[pos];
    }
};


template<typename T, MmaType type = MmaType::kLegacy>
struct WarpMma {
public:
    using ThreadScaleT          = ThreadScale<T, 4>;
    using FragmentSrcAIteratorT = FragmentSrcIterator<T, true>;
    using FragmentSrcBIteratorT = FragmentSrcIterator<T, false>;
    using FragmentResIteratorT  = FragmentResIterator<T>;
    using ThreadGemmShapeT = GemmShape<4, 4, 1>;
    using ThreadMmaT       = ThreadMma<T, ThreadGemmShapeT>;

public:
    COGITO_DEVICE
    void operator()(const T& alpha, FragmentSrcAIteratorT& frag_a, FragmentSrcBIteratorT& frag_b, const T& beta, FragmentResIteratorT& frag_c) {
        
        ThreadMmaT mma_op;
        ThreadScaleT scale_op;

        for (int i = 0; i < FragmentSrcAIteratorT::kLoop; ++i) {

            frag_a.load();
            frag_b.load();

            scale_op(alpha, frag_a[0]);
            scale_op(alpha, frag_a[1]);

            mma_op(frag_a[0], frag_b[0], frag_c[0]);
            mma_op(frag_a[0], frag_b[1], frag_c[1]);
            mma_op(frag_a[1], frag_b[0], frag_c[2]);
            mma_op(frag_a[1], frag_b[1], frag_c[3]);

            frag_a++;
            frag_b++;
        }
    }
};

/*
template<>
struct WarpMma<half_t, MmaType::kTensorCore> {
public:
    static constexpr int kM = 16;
    static constexpr int kN = 16;
    static constexpr int kK = 16;
    using FragmentSrcAIteratorT = typename nvcuda::wmma::fragment template<typename nvcuda::wmma::matrix_a, kM, kN, kK, half_t, typename nvcuda::wmma::row_major>;
    using FragmentSrcBIteratorT = typename nvcuda::wmma::fragment template<typename nvcuda::wmma::matrix_b, kM, kN, kK, half_t, typename nvcuda::wmma::row_major>;
    using FragmentResIteratorT  = typename nvcuda::wmma::fragment template<typename nvcuda::wmma::accumulator, kM, kN, kK, half_t>;

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
