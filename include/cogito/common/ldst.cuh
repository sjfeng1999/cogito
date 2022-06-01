//
//
// 
//

#pragma once 

#include <cstdint>

#include "cogito/tensor.cuh"
#include "cogito/ptx.cuh"

namespace cogito {

///////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, LoadCachePolicy LdPolicy = LoadCachePolicy::kDefault, StoreCachePolicy StPolicy = StoreCachePolicy::kDefault>
struct ThreadLdSt {
public:
    static constexpr int kElementSize = sizeof(T);
    static_assert(mp::IsPow2<kElementSize>::value, "Invalid element size");
    static constexpr LoadCachePolicy  kLdPolicy = LdPolicy;
    static constexpr StoreCachePolicy kStPolicy = StPolicy;

public:
    // Load by LDG/S.128
    template<int TensorSize, int Start = 0, int Length = TensorSize, 
        typename std::enable_if<(Start  * kElementSize % 16 == 0) && 
                                (Length * kElementSize % 16 == 0), int>::type Factor = (Length * kElementSize >> 4)>
    COGITO_DEVICE
    static void load(ShapedTensor<T, TensorSize>& tensor, const T* ptr) {
        static_assert(Start + Length <= TensorSize, "Load size exceed tensor size");
        COGITO_PRAGMA_UNROLL
        for (int i = 0; i < Factor; ++i) {
            // TODO (load according to cache policy)
            float4 val = *reinterpret_cast<const float4*>(reinterpret_cast<const int8_t*>(ptr) + i * 16);
            *reinterpret_cast<float4*>(reinterpret_cast<int8_t*>(&tensor[Start]) + i * 16) = val;
        }
    }

    // Load by LDG/S.64
    template<int TensorSize, int Start = 0, int Length = TensorSize, 
        typename std::enable_if<(Length * kElementSize % 16 != 0 || Start * kElementSize % 16 != 0) &&
                                (Start  * kElementSize %  8 == 0) &&
                                (Length * kElementSize %  8 == 0), int>::type Factor = (Length * kElementSize >> 3)>
    COGITO_DEVICE
    static void load(ShapedTensor<T, TensorSize>& tensor, const T* ptr) {
        static_assert(Start + Length <= TensorSize, "Load size exceed tensor size");
        COGITO_PRAGMA_UNROLL
        for (int i = 0; i < Factor; ++i) {
            float2 val = *reinterpret_cast<const float2*>(reinterpret_cast<const int8_t*>(ptr) + i * 8);
            *reinterpret_cast<float2*>(reinterpret_cast<int8_t*>(&tensor[Start]) + i * 8) = val;
        }
    }
 
    template<int TensorSize, int Start = 0, int Length = TensorSize, 
        typename std::enable_if<(Length * kElementSize % 16 != 0 || Start * kElementSize % 16 != 0) &&
                                (Length * kElementSize %  8 != 0 || Start * kElementSize %  8 != 0), int>::type = 0>
    COGITO_DEVICE
    static void load(ShapedTensor<T, TensorSize>& tensor, const T* ptr) {
        static_assert(Start + Length <= TensorSize, "Load size exceed tensor size");
        COGITO_PRAGMA_UNROLL
        for (int i = 0; i < Length; ++i) {
            tensor[i + Start] = ptr[i];
        }
    }

    template<int TensorSize, int Start = 0, int Length = TensorSize>
    COGITO_DEVICE
    static void load(ShapedTensor<T, TensorSize>& tensor, const T const_val) {
        static_assert(Start + Length <= TensorSize, "Load size exceed tensor size");
        COGITO_PRAGMA_UNROLL
        for (int i = Start; i < Start + Length; ++i) {
            tensor[i] = const_val;
        }
    }


    template<int Start, int BlockSize, int LineSize, int TensorSize, int RangeStart, int RangeEnd>
    COGITO_DEVICE
    static void stripedLoad(ShapedTensor<T, TensorSize>& tensor, const T* ptr, mp::Range2Type<RangeStart, RangeEnd> /* unused */) {
        static_assert(Start + (RangeEnd - RangeStart - 1) * BlockSize < TensorSize);
        ThreadLdSt::load<TensorSize, Start + RangeStart * BlockSize, BlockSize>(tensor, ptr);
        ThreadLdSt::stripedLoad<Start, BlockSize, LineSize>(tensor, ptr + LineSize, mp::Range2Type<RangeStart + 1, RangeEnd>{});
    }
    template<int Start, int BlockSize, int LineSize, int TensorSize, int RangeEnd>
    COGITO_DEVICE
    static void stripedLoad(ShapedTensor<T, TensorSize>& tensor, const T* ptr, mp::Range2Type<RangeEnd, RangeEnd> /* unused */) {}

    template<int Start, int BlockSize, int TensorSize, int RangeStart, int RangeEnd>
    COGITO_DEVICE
    static void stripedLoad(ShapedTensor<T, TensorSize>& tensor, const T* ptr, const int ldg, mp::Range2Type<RangeStart, RangeEnd> /* unused */) {
        static_assert(Start + (RangeEnd - RangeStart - 1) * BlockSize < TensorSize);
        ThreadLdSt::load<TensorSize, Start + RangeStart * BlockSize, BlockSize>(tensor, ptr);
        ThreadLdSt::stripedLoad<Start, BlockSize>(tensor, ptr + ldg, ldg, mp::Range2Type<RangeStart + 1, RangeEnd>{});
    }
    template<int Start, int BlockSize, int TensorSize, int RangeEnd>
    COGITO_DEVICE
    static void stripedLoad(ShapedTensor<T, TensorSize>&, const T*, const int, mp::Range2Type<RangeEnd, RangeEnd> /* unused */) {}

    ///////////////////////////////////////////////////////////////////////////////////////////////

    // Store by STG/S.128
    template<int TensorSize, int Start = 0, int Length = TensorSize, 
        typename std::enable_if<(Start  * kElementSize % 16 == 0) && 
                                (Length * kElementSize % 16 == 0), int>::type Factor = (Length * kElementSize >> 4)>
    COGITO_DEVICE
    static void store(const ShapedTensor<T, TensorSize>& tensor, T* ptr) {
        static_assert(Start + Length <= TensorSize, "Load size exceed tensor size");
        COGITO_PRAGMA_UNROLL
        for (int i = 0; i < Factor; ++i) {
            float4 val = *reinterpret_cast<const float4*>(reinterpret_cast<const int8_t*>(&tensor[Start]) + i * 16);
            *reinterpret_cast<float4*>(reinterpret_cast<int8_t*>(ptr) + i * 16) = val;
        } 
    }

    // Store by STG/S.64
    template<int TensorSize, int Start = 0, int Length = TensorSize, 
        typename std::enable_if<(Length * kElementSize % 16 != 0 || Start * kElementSize % 16 != 0) &&
                                (Start  * kElementSize %  8 == 0) &&
                                (Length * kElementSize %  8 == 0), int>::type Factor = (Length * kElementSize >> 3)>
    COGITO_DEVICE
    static void store(const ShapedTensor<T, TensorSize>& tensor, T* ptr) {
        static_assert(Start + Length <= TensorSize, "Load size exceed tensor size");
        COGITO_PRAGMA_UNROLL
        for (int i = 0; i < Factor; ++i) {
            float2 val = *reinterpret_cast<const float2*>(reinterpret_cast<const int8_t*>(&tensor[Start]) + i * 8);
            *reinterpret_cast<float2*>(reinterpret_cast<int8_t*>(ptr) + i * 8) = val;
        }
    }

    template<int TensorSize, int Start = 0, int Length = TensorSize, 
        typename std::enable_if<(Length * kElementSize % 16 != 0 || Start * kElementSize % 16 != 0) &&
                                (Length * kElementSize %  8 != 0 || Start * kElementSize %  8 != 0), int>::type = 0>
    COGITO_DEVICE
    static void store(const ShapedTensor<T, TensorSize>& tensor, T* ptr) {
        static_assert(Start + Length <= TensorSize, "Load size exceed tensor size");
        COGITO_PRAGMA_UNROLL
        for (int i = 0; i < Length; ++i) {
            ptr[i] = tensor[Start + i];
        }
    }


    template<int Start, int BlockSize, int LineSize, int TensorSize, int RangeStart, int RangeEnd>
    COGITO_DEVICE
    static void stripedStore(const ShapedTensor<T, TensorSize>& tensor, T* ptr, mp::Range2Type<RangeStart, RangeEnd> /* unused */) {
        static_assert(Start + (RangeEnd - RangeStart - 1) * BlockSize < TensorSize);
        ThreadLdSt::store<TensorSize, Start + RangeStart * BlockSize, BlockSize>(tensor, ptr);
        ThreadLdSt::stripedStore<Start, BlockSize, LineSize>(tensor, ptr + LineSize, mp::Range2Type<RangeStart + 1, RangeEnd>{});
    }
    template<int Start, int BlockSize, int LineSize, int TensorSize, int RangeEnd>
    COGITO_DEVICE
    static void stripedStore(const ShapedTensor<T, TensorSize>&, T*, mp::Range2Type<RangeEnd, RangeEnd> /* unused */) {}

    template<int Start, int BlockSize, int TensorSize, int RangeStart, int RangeEnd>
    COGITO_DEVICE
    static void stripedStore(const ShapedTensor<T, TensorSize>& tensor, T* ptr, const int ldg, mp::Range2Type<RangeStart, RangeEnd> /* unused */) {
        static_assert(Start + (RangeEnd - RangeStart - 1) * BlockSize < TensorSize);
        ThreadLdSt::store<TensorSize, Start + RangeStart * BlockSize, BlockSize>(tensor, ptr);
        ThreadLdSt::stripedStore<Start, BlockSize>(tensor, ptr + ldg, ldg, mp::Range2Type<RangeStart + 1, RangeEnd>{});
    }
    template<int Start, int BlockSize, int TensorSize, int RangeEnd>
    COGITO_DEVICE
    static void stripedStore(const ShapedTensor<T, TensorSize>&, T*, const int, mp::Range2Type<RangeEnd, RangeEnd> /* unused */) {}
};

///////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, LoadCachePolicy LdPolicy = LoadCachePolicy::kDefault, StoreCachePolicy StPolicy = StoreCachePolicy::kDefault>
struct WarpLdSt {
public:
    static constexpr int kElementSize = sizeof(T);
    static_assert(mp::IsPow2<kElementSize>::value, "Invalid element size");
    static constexpr LoadCachePolicy  kLdPolicy = LdPolicy;
    static constexpr StoreCachePolicy kStPolicy = StPolicy;

public:
    template<int Start, int BlockSize, int TensorSize>
    COGITO_DEVICE
    static void load(ShapedTensor<T, TensorSize>& tensor, const T* ptr, const uint32_t mask) {
        int laneid = ptx::getLaneid();
        if (((1u << laneid) & mask) != 0) {
            ThreadLdSt<T, kLdPolicy, kStPolicy>::load<TensorSize, Start, BlockSize>(tensor, ptr + laneid * BlockSize);
        }
    }

    template<int Start, int BlockSize, int TensorSize>
    COGITO_DEVICE
    static void load(ShapedTensor<T, TensorSize>& tensor, const T* ptr) {
        int laneid = ptx::getLaneid();
        ThreadLdSt<T, kLdPolicy, kStPolicy>::load<TensorSize, Start, BlockSize>(tensor, ptr + laneid * BlockSize);
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////

    template<int Start, int BlockSize, int TensorSize>
    COGITO_DEVICE
    static void store(const ShapedTensor<T, TensorSize>& tensor, T* ptr, const uint32_t mask) {
        int laneid = ptx::getLaneid();
        if (((1u << laneid) & mask) != 0) {
            ThreadLdSt<T, kLdPolicy, kStPolicy>::store<TensorSize, Start, BlockSize>(tensor, ptr + laneid * BlockSize);
        }
    }

    template<int Start, int BlockSize, int TensorSize>
    COGITO_DEVICE
    static void store(const ShapedTensor<T, TensorSize>& tensor, T* ptr) {
        int laneid = ptx::getLaneid();
        ThreadLdSt<T, kLdPolicy, kStPolicy>::store<TensorSize, Start, BlockSize>(tensor, ptr + laneid * BlockSize);
    }
};

// ///////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, int GroupSize, LoadCachePolicy LdPolicy = LoadCachePolicy::kDefault, StoreCachePolicy StPolicy = StoreCachePolicy::kDefault>
struct ThreadGroupLdSt {
public:
    static constexpr int kGroupSize   = GroupSize;
    static constexpr int kElementSize = sizeof(T);
    static_assert(mp::IsPow2<kElementSize>::value, "Invalid element size");
    static constexpr LoadCachePolicy  kLdPolicy = LdPolicy;
    static constexpr StoreCachePolicy kStPolicy = StPolicy;

public:
    template<int Start, int BlockSize, int TensorSize, int RangeStart, int RangeEnd>
    COGITO_DEVICE
    static void load(ShapedTensor<T, TensorSize>& tensor, const T* ptr, mp::Range2Type<RangeStart, RangeEnd> /* unused */) {
        int group_id = threadIdx.x % kGroupSize;
        ThreadLdSt<T, kLdPolicy, kStPolicy>::stripedLoad<Start, BlockSize, BlockSize * kGroupSize>(tensor, ptr + group_id * BlockSize, mp::Range2Type<RangeStart, RangeEnd>{});
    }
    template<int Start, int BlockSize, int TensorSize, int RangeEnd>
    COGITO_DEVICE
    static void load(ShapedTensor<T, TensorSize>& tensor, const T* ptr, mp::Range2Type<RangeEnd, RangeEnd> /* unused */) {}

    ///////////////////////////////////////////////////////////////////////////////////////////////

    template<int Start, int BlockSize, int TensorSize, int RangeStart, int RangeEnd>
    COGITO_DEVICE
    static void store(const ShapedTensor<T, TensorSize>& tensor, T* ptr, mp::Range2Type<RangeStart, RangeEnd> /* unused */) {
        int group_id = threadIdx.x % kGroupSize;
        ThreadLdSt<T, kLdPolicy, kStPolicy>::stripedStore<Start, BlockSize, BlockSize * kGroupSize>(tensor, ptr + group_id * BlockSize, mp::Range2Type<RangeStart, RangeEnd>{});
    }
    template<int Start, int BlockSize, int TensorSize, int RangeEnd>
    COGITO_DEVICE
    static void store(const ShapedTensor<T, TensorSize>& tensor, T* ptr, mp::Range2Type<RangeEnd, RangeEnd> /* unused */) {}
};

} // namespace cogito

