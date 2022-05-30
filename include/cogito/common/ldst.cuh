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
    static_assert(mp::IsPow2<kElementSize>::value, "Invalid element size");
    static constexpr int kBlockSize      = blockSize;
    static constexpr int kStripSize      = stripSize;
    static constexpr int kItemsPerThread = kBlockSize * kStripSize;

public:
    // Load by LDG/S.128
    template<int Start = 0, int Length = kItemsPerThread, 
        typename std::enable_if<Length * kElementSize % 16 == 0, int>::type Factor = (Length * kElementSize >> 4)>
    COGITO_DEVICE
    static void load(ShapedTensor<T, kItemsPerThread>& tensor, const T* ptr, bool valid) {
        static_assert(Start + Length <= kItemsPerThread, "Load size exceed tensor size");

        if (valid) {
            COGITO_PRAGMA_UNROLL
            for (int i = 0; i < Factor; ++i) {
                float4 val = *reinterpret_cast<const float4*>(reinterpret_cast<const int8_t*>(ptr) + i * 16);
                *reinterpret_cast<float4*>(reinterpret_cast<int8_t*>(&tensor[Start]) + i * 16) = val;
            }
        }
    }

    // Load by LDG/S.64
    template<int Start = 0, int Length = kItemsPerThread, 
        typename std::enable_if<(Length * kElementSize % 16 != 0) && (Length * kElementSize % 8 == 0), int>::type Factor = (Length * kElementSize >> 3)>
    COGITO_DEVICE
    static void load(ShapedTensor<T, kItemsPerThread>& tensor, const T* ptr, bool valid) {
        static_assert(Start + Length <= kItemsPerThread, "Load size exceed tensor size");

        if (valid) {
            COGITO_PRAGMA_UNROLL
            for (int i = 0; i < Factor; ++i) {
                float2 val = *reinterpret_cast<const float2*>(reinterpret_cast<const int8_t*>(ptr) + i * 8);
                *reinterpret_cast<float2*>(reinterpret_cast<int8_t*>(&tensor[Start]) + i * 8) = val;
            }
        }
    }

    template<int Start = 0, int Length = kItemsPerThread, 
        typename std::enable_if<(Length * kElementSize % 16 != 0) && (Length * kElementSize % 8 != 0), int>::type = 0>
    COGITO_DEVICE
    static void load(ShapedTensor<T, kItemsPerThread>& tensor, const T* ptr, bool valid) {
        static_assert(Start + Length <= kItemsPerThread, "Load size exceed tensor size");

        if (valid) {
            COGITO_PRAGMA_UNROLL
            for (int i = Start; i < Start + Length; ++i) {
                tensor[i] = ptr[i];
            }
        }
    }

    template<int Start = 0, int Length = kItemsPerThread>
    COGITO_DEVICE
    static void load(ShapedTensor<T, kItemsPerThread>& tensor, const T& const_val, bool valid) {
        static_assert(Start + Length <= kItemsPerThread, "Load size exceed tensor size");

        if (valid) {
            COGITO_PRAGMA_UNROLL
            for (int i = Start; i < Start + Length; ++i) {
                tensor[i] = const_val;
            }
        }
    }

    template<int LineSize, int rangeStart, int rangeEnd>
    COGITO_DEVICE
    static void stripedLoad(ShapedTensor<T, kItemsPerThread>& tensor, const T* ptr, mp::Range2Type<rangeStart, rangeEnd> /* unused */) {
        ThreadLdSt::load<rangeStart * kBlockSize, kBlockSize>(tensor, ptr, true);
        ThreadLdSt::stripedLoad<LineSize>(tensor, ptr + LineSize, mp::Range2Type<rangeStart + 1, rangeEnd>{});
    }

    template<int LineSize, int rangeEnd>
    COGITO_DEVICE
    static void stripedLoad(ShapedTensor<T, kItemsPerThread>& tensor, const T* ptr, mp::Range2Type<rangeEnd, rangeEnd /* unused */>) {}

    ///////////////////////////////////////////////////////////////////////////////////////////////

    // Store by STG/S.128
    template<int Start = 0, int Length = kItemsPerThread, 
        typename std::enable_if<Length * kElementSize % 16 == 0, int>::type Factor = (Length * kElementSize >> 4)>
    COGITO_DEVICE
    static void store(const ShapedTensor<T, kItemsPerThread>& tensor, T* ptr, bool valid) {
        static_assert(Start + Length <= kItemsPerThread, "Load size exceed tensor size");

        if (valid) {
            COGITO_PRAGMA_UNROLL
            for (int i = 0; i < Factor; ++i) {
                float4 val = *reinterpret_cast<const float4*>(reinterpret_cast<const int8_t*>(&tensor[Start]) + i * 16);
                *reinterpret_cast<float4*>(reinterpret_cast<int8_t*>(ptr) + i * 16) = val;
            } 
        }
    }

    // Store by STG/S.64
    template<int Start = 0, int Length = kItemsPerThread, 
        typename std::enable_if<(Length * kElementSize % 16 != 0) && (Length * kElementSize % 8 == 0), int>::type Factor = (Length * kElementSize >> 3)>
    COGITO_DEVICE
    static void store(const ShapedTensor<T, kItemsPerThread>& tensor, T* ptr, bool valid) {
        static_assert(Start + Length <= kItemsPerThread, "Load size exceed tensor size");

        if (valid) {
            COGITO_PRAGMA_UNROLL
            for (int i = 0; i < Factor; ++i) {
                float2 val = *reinterpret_cast<const float2*>(reinterpret_cast<const int8_t*>(&tensor[Start]) + i * 8);
                *reinterpret_cast<float2*>(reinterpret_cast<int8_t*>(ptr) + i * 8) = val;
            }
        }
    }

    template<int Start = 0, int Length = kItemsPerThread, 
        typename std::enable_if<(Length * kElementSize % 16 != 0) && (Length * kElementSize % 8 != 0), int>::type = 0>
    COGITO_DEVICE
    static void store(const ShapedTensor<T, kItemsPerThread>& tensor, T* ptr, bool valid) {
        static_assert(Start + Length <= kItemsPerThread, "Load size exceed tensor size");

        if (valid) {
            COGITO_PRAGMA_UNROLL
            for (int i = Start; i < Start + Length; ++i) {
                ptr[i] = tensor[i];
            }
        }
    }

    template<int LineSize, int rangeStart, int rangeEnd>
    COGITO_DEVICE
    static void stripedStore(const ShapedTensor<T, kItemsPerThread>& tensor, T* ptr, mp::Range2Type<rangeStart, rangeEnd> /* unused */) {
        ThreadLdSt::store<rangeStart * kBlockSize, kBlockSize>(tensor, ptr, true);
        ThreadLdSt::stripedStore<LineSize>(tensor, ptr + LineSize, mp::Range2Type<rangeStart + 1, rangeEnd>{});
    }

    template<int LineSize, int rangeEnd>
    COGITO_DEVICE
    static void stripedStore(const ShapedTensor<T, kItemsPerThread>& tensor, T* ptr, mp::Range2Type<rangeEnd, rangeEnd> /* unused */) {}
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

