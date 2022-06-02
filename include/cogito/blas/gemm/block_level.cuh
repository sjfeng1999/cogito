//
// 
//
//

#pragma once 

#include "cogito/blas/gemm/warp_level.cuh"

namespace cogito {
namespace blas {
namespace detail {

///////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, bool Transpose, MmaType mma_type>
struct TileSrcIterator { 
public:
    static constexpr int kTileHeight = 8;
    static constexpr int kTileWidth  = 128;
    static constexpr int kSharedSize = kTileHeight * kTileWidth;
    static constexpr bool kTranspose = Transpose;

private:
    cogito_shared_ptr T* const shared_ptr_;
    cogito_shared_ptr T* shared_ptr_offset_;
    cogito_device_ptr T* global_ptr_offset_;
    cogito_device_reg ShapedTensor<T, 4> frag_;
    const int ldg_;
    int loop;
    int total_loop;

public:
    TileSrcIterator() = delete;
    
    COGITO_DEVICE
    TileSrcIterator(T* global_ptr, const int ldg, T* shared_ptr) : ldg_(ldg), shared_ptr_(shared_ptr), loop(0) { 
        total_loop = ldg_ / kTileHeight;
        {   
            int tid = threadIdx.x;
            int tile_x, tile_y;

            if (kTranspose) {
                tile_x = tid & 0x1;
                tile_y = tid >> 1;
                global_ptr_offset_ = global_ptr + (tile_x << 2) + tile_y * ldg_;
                shared_ptr_offset_ = shared_ptr + tile_y + tile_x * kTileWidth * 4;
            } else {
                tile_x = tid & 0x1f;
                tile_y = tid >> 5;
                global_ptr_offset_ = global_ptr + (tile_x << 2) + tile_y * ldg_;
                shared_ptr_offset_ = shared_ptr + (tile_x << 2) + tile_y * kTileWidth;
            }
        }
        update(); 
        loop++;
    }

    COGITO_DEVICE
    void update() {
        ThreadLdSt<T>::load(frag_, global_ptr_offset_);
        if (kTranspose) {             
            ThreadLdSt<T>::stripedStore<0, 1, kTileWidth>(frag_, shared_ptr_offset_, mp::Range2Type<0, 4>{});
        } else {
            ThreadLdSt<T>::store(frag_, shared_ptr_offset_);
        }
    }

    COGITO_DEVICE
    void operator++(int) {
        if (kTranspose){
            global_ptr_offset_ += kTileHeight;
        } else {
            global_ptr_offset_ += kTileHeight * ldg_;
        }
        update();
        loop++;
    }

    COGITO_DEVICE
    T* shared_ptr() const { return shared_ptr_; }

    COGITO_DEVICE
    bool end() { return loop == total_loop ? true : false; }
};


template<typename T, MmaType mma_type>
struct TileResIterator {
private:
    cogito_device_ptr T* const global_ptr_;
    const int ldg_;

public:
    TileResIterator() = delete;

    COGITO_DEVICE
    TileResIterator(T* global_ptr, const int ldg) : global_ptr_(global_ptr), ldg_(ldg) {}

    COGITO_DEVICE
    T* global_ptr() const { return global_ptr_; }

    COGITO_DEVICE
    int ldg() const { return ldg_; }
};


template<typename T, MmaType mma_type>
struct BlockMma {
public:
    static constexpr int kM = 0;
    static constexpr int kN = 0;
    static constexpr int kK = 0;

    using TileSrcAIteratorT = TileSrcIterator<T, true, mma_type>;
    using TileSrcBIteratorT = TileSrcIterator<T, false, mma_type>;
    using TileResIteratorT  = TileResIterator<T, mma_type>;

    using WarpMmaT = WarpMma<T, mma_type>;
    using FragmentSrcAIteratorT = typename WarpMmaT::FragmentSrcAIteratorT;
    using FragmentSrcBIteratorT = typename WarpMmaT::FragmentSrcBIteratorT;
    using FragmentResIteratorT  = typename WarpMmaT::FragmentResIteratorT;

public:
    COGITO_DEVICE
    void operator()(T alpha, TileSrcAIteratorT& tile_a, TileSrcBIteratorT& tile_b, T beta, TileResIteratorT& tile_c){

        FragmentSrcAIteratorT frag_a(tile_a.shared_ptr());
        FragmentSrcBIteratorT frag_b(tile_b.shared_ptr());
        FragmentResIteratorT  frag_c(beta, tile_c.global_ptr(), tile_c.ldg());
        __syncthreads();

        WarpMmaT op;
        op(alpha, frag_a, frag_b, beta, frag_c);
        __syncthreads();

        while (!tile_a.end()){
            tile_a++;
            tile_b++;
            frag_a.reset(tile_a.shared_ptr());
            frag_b.reset(tile_b.shared_ptr());
            __syncthreads();

            op(alpha, frag_a, frag_b, beta, frag_c);
            __syncthreads();
        }
        frag_c.store();
    }
};

} // namespace detail
} // namespace blas
} // namespace cogito
