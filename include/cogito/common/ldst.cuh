//
//
// 
//

#pragma once 

#include <cstdint>

#include "cogito/cogito.cuh"
#include "cogito/tensor.cuh"

namespace cogito {

///////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, int blockSize, int stripSize = 1>
struct ThreadLdSt {
public:
    static constexpr int kElementSize    = sizeof(T);
    static_assert(mp::IsPow2<kElementSize>::value);
    static constexpr int kItemsPer4B     =  4 / sizeof(T);
    static constexpr int kItemsPer8B     =  8 / sizeof(T);
    static constexpr int kItemsPer16B    = 16 / sizeof(T);
    static constexpr int kBlockSize      = blockSize;
    static constexpr int kStripSize      = stripSize;
    static constexpr int kItemsPerThread = kBlockSize * kStripSize;

public:
    // Load by LDG/S.128
    template<int Start = 0, int Length = kItemsPerThread, 
        typename std::enable_if<Length * kElementSize % 16 == 0, int>::type Factor = (Length * kElementSize >> 4)>
    COGITO_DEVICE
    static void load(ShapedTensor<T, kItemsPerThread>& tensor, const T* ptr, bool valid) {
        if (valid) {
            COGITO_PRAGMA_UNROLL
            for (int i = 0; i < Factor; ++i) {
                float4 val = *reinterpret_cast<const float4*>(reinterpret_cast<const int8_t*>(ptr) + i * 16);
                tensor[Start + i * kItemsPer16B + 0 * kItemsPer4B] = reinterpret_cast<T&>(val.x);
                tensor[Start + i * kItemsPer16B + 1 * kItemsPer4B] = reinterpret_cast<T&>(val.y);
                tensor[Start + i * kItemsPer16B + 2 * kItemsPer4B] = reinterpret_cast<T&>(val.z);
                tensor[Start + i * kItemsPer16B + 3 * kItemsPer4B] = reinterpret_cast<T&>(val.w);
            }
        }
    }

    // Load by LDG/S.64
    template<int Start = 0, int Length = kItemsPerThread, 
        typename std::enable_if<(Length * kElementSize % 16 != 0) && (Length * kElementSize % 8 == 0), int>::type Factor = (Length * kElementSize >> 3)>
    COGITO_DEVICE
    static void load(ShapedTensor<T, kItemsPerThread>& tensor, const T* ptr, bool valid) {
        if (valid) {
            COGITO_PRAGMA_UNROLL
            for (int i = 0; i < Factor; ++i) {
                float2 val = *reinterpret_cast<const float2*>(reinterpret_cast<const int8_t*>(ptr) + i * 8);
                tensor[Start + i * kItemsPer8B + 0 * kItemsPer4B] = reinterpret_cast<T&>(val.x);
                tensor[Start + i * kItemsPer8B + 1 * kItemsPer4B] = reinterpret_cast<T&>(val.y);
            }
        }
    }

    template<int Start = 0, int Length = kItemsPerThread, 
        typename std::enable_if<(Length * kElementSize % 16 != 0) && (Length * kElementSize % 8 != 0), int>::type = 0>
    COGITO_DEVICE
    static void load(ShapedTensor<T, kItemsPerThread>& tensor, const T* ptr, bool valid) {
        if (valid) {
            COGITO_PRAGMA_UNROLL
            for (int i = Start; i < Start + Length; ++i) {
                tensor[i] = ptr[i];
            }
        }
    }

    template<int LineSize, int rangeStart, int rangeEnd>
    COGITO_DEVICE
    static void stripedLoad(ShapedTensor<T, kItemsPerThread>& tensor, const T* ptr, mp::Range2Type<rangeStart, rangeEnd>) {
        ThreadLdSt::load<rangeStart * kBlockSize, kBlockSize>(tensor, ptr, true);
        ThreadLdSt::stripedLoad<LineSize>(tensor, ptr + LineSize, mp::Range2Type<rangeStart + 1, rangeEnd>{});
    }

    template<int LineSize, int rangeEnd>
    COGITO_DEVICE
    static void stripedLoad(ShapedTensor<T, kItemsPerThread>& tensor, const T* ptr, mp::Range2Type<rangeEnd, rangeEnd>) {}

    ///////////////////////////////////////////////////////////////////////////////////////////////

    // Store by STG/S.128
    template<int Start = 0, int Length = kItemsPerThread, 
        typename std::enable_if<Length * kElementSize % 16 == 0, int>::type Factor = (Length * kElementSize >> 4)>
    COGITO_DEVICE
    static void store(const ShapedTensor<T, kItemsPerThread>& tensor, T* ptr, bool valid) {
        if (valid) {
            COGITO_PRAGMA_UNROLL
            for (int i = 0; i < Factor; ++i) {
                float4 val;
                val.x = reinterpret_cast<const T&>(tensor[Start + i * kItemsPer16B + 0 * kItemsPer4B]);
                val.y = reinterpret_cast<const T&>(tensor[Start + i * kItemsPer16B + 1 * kItemsPer4B]);
                val.z = reinterpret_cast<const T&>(tensor[Start + i * kItemsPer16B + 2 * kItemsPer4B]);
                val.w = reinterpret_cast<const T&>(tensor[Start + i * kItemsPer16B + 3 * kItemsPer4B]);
                *reinterpret_cast<float4*>(reinterpret_cast<int8_t*>(ptr) + i * 16) = val;
            } 
        }
    }

    // Store by STG/S.64
    template<int Start = 0, int Length = kItemsPerThread, 
        typename std::enable_if<(Length * kElementSize % 16 != 0) && (Length * kElementSize % 8 == 0), int>::type Factor = (Length * kElementSize >> 3)>
    COGITO_DEVICE
    static void store(const ShapedTensor<T, kItemsPerThread>& tensor, T* ptr, bool valid) {
        if (valid) {
            COGITO_PRAGMA_UNROLL
            for (int i = 0; i < Factor; ++i) {
                float2 val;
                val.x = reinterpret_cast<const T&>(tensor[Start + i * kItemsPer8B + 0 * kItemsPer4B]);
                val.y = reinterpret_cast<const T&>(tensor[Start + i * kItemsPer8B + 1 * kItemsPer4B]);
                *reinterpret_cast<float2*>(reinterpret_cast<int8_t*>(ptr) + i * 8) = val;
            }
        }
    }

    template<int Start = 0, int Length = kItemsPerThread, 
        typename std::enable_if<(Length * kElementSize % 16 != 0) && (Length * kElementSize % 8 != 0), int>::type = 0>
    COGITO_DEVICE
    static void store(const ShapedTensor<T, kItemsPerThread>& tensor, T* ptr, bool valid) {
        if (valid) {
            COGITO_PRAGMA_UNROLL
            for (int i = Start; i < Start + Length; ++i) {
                ptr[i] = tensor[i];
            }
        }
    }


    template<int LineSize, int rangeStart, int rangeEnd>
    COGITO_DEVICE
    static void stripedStore(const ShapedTensor<T, kItemsPerThread>& tensor, T* ptr, mp::Range2Type<rangeStart, rangeEnd>) {
        ThreadLdSt::store<rangeStart * kBlockSize, kBlockSize>(tensor, ptr, true);
        ThreadLdSt::stripedStore<LineSize>(tensor, ptr + LineSize, mp::Range2Type<rangeStart + 1, rangeEnd>{});
    }

    template<int LineSize, int rangeEnd>
    COGITO_DEVICE
    static void stripedStore(const ShapedTensor<T, kItemsPerThread>& tensor, T* ptr, mp::Range2Type<rangeEnd, rangeEnd>) {}
};

///////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, int blockSize, int stripSize = 1>
struct WarpLdSt {
public:
    static constexpr int kBlockSize      = blockSize;
    static constexpr int kStripSize      = stripSize;
    static constexpr int kWarpLineSize   = cogito::kWarpSize * kBlockSize;
    static constexpr int kItemsPerThread = kBlockSize * kStripSize;

public:
    COGITO_DEVICE
    static void load(ShapedTensor<T, kItemsPerThread>& tensor, const T* ptr, const uint32_t& mask) {
        uint32_t tid = threadIdx.x;
        if (((1u << tid) & mask) != 0) {
            ThreadLdSt<T, kBlockSize, kStripSize>::stripedLoad<kWarpLineSize>(tensor, ptr + tid * kBlockSize, mp::Range2Type<0, kStripSize>{});
        }
    }

    COGITO_DEVICE
    static void load(ShapedTensor<T, kItemsPerThread>& tensor, const T* ptr) {
        uint32_t tid = threadIdx.x;
        ThreadLdSt<T, kBlockSize, kStripSize>::stripedLoad<kWarpLineSize>(tensor, ptr + tid * kBlockSize, mp::Range2Type<0, kStripSize>{});
    }

    COGITO_DEVICE
    static void store(const ShapedTensor<T, kItemsPerThread>& tensor, T* ptr, const uint32_t& mask) {
        uint32_t tid = threadIdx.x;
        if (((1u << tid) & mask) != 0) {
            ThreadLdSt<T, kBlockSize, kStripSize>::stripedStore<kWarpLineSize>(tensor, ptr + tid * kBlockSize, mp::Range2Type<0, kStripSize>{});
        }
    }

    COGITO_DEVICE
    static void store(const ShapedTensor<T, kItemsPerThread>& tensor, T* ptr) {
        uint32_t tid = threadIdx.x;
        ThreadLdSt<T, kBlockSize, kStripSize>::stripedStore<kWarpLineSize>(tensor, ptr + tid * kBlockSize, mp::Range2Type<0, kStripSize>{});
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, int GroupSize, int blockSize, int stripSize = 1>
struct ThreadGroupLdSt {
public:
    static constexpr int kGroupSize      = GroupSize;
    static constexpr int kBlockSize      = blockSize;
    static constexpr int kStripSize      = stripSize;
    static constexpr int kGroupLineSize  = kGroupSize * kBlockSize;
    static constexpr int kItemsPerThread = kBlockSize * kStripSize;

public:
    COGITO_DEVICE
    static void load(ShapedTensor<T, kItemsPerThread>& tensor, const T* ptr, bool valid) {
        if (valid) {
            int tid = threadIdx.x;
            ThreadLdSt<T, kBlockSize, kStripSize>::stripedLoad<kGroupLineSize>(tensor, ptr + tid * kBlockSize, mp::Range2Type<0, kStripSize>{});
        }
    }

    COGITO_DEVICE
    static void store(const ShapedTensor<T, kItemsPerThread>& tensor, T* ptr, bool valid) {
        if (valid) {
            int tid = threadIdx.x;
            ThreadLdSt<T, kBlockSize, kStripSize>::stripedStore<kGroupLineSize>(tensor, ptr + tid * kBlockSize, mp::Range2Type<0, kStripSize>{});
        }
    }
};

} // namespace cogito

