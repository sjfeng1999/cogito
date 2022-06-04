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

template<typename T, LoadPolicy LdPolicy = LoadPolicy::kDefault>
struct ThreadLd {
public:
    static constexpr int kElementSize = sizeof(T);
    static_assert(mp::IsPow2<kElementSize>::value, "Invalid element size");
    static constexpr LoadPolicy  kLdPolicy = LdPolicy;

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
            ptx::ld_128b<kLdPolicy>(static_cast<void*>(reinterpret_cast<int8_t*>(&tensor[Start]) + i * 16), 
                static_cast<void*>(reinterpret_cast<int8_t*>(const_cast<T*>(ptr)) + i * 16));
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
            ptx::ld_64b<kLdPolicy>(static_cast<void*>(reinterpret_cast<int8_t*>(&tensor[Start]) + i * 8), 
                static_cast<void*>(reinterpret_cast<int8_t*>(const_cast<T*>(ptr)) + i * 8));
        }
    }
 
    template<int TensorSize, int Start = 0, int Length = TensorSize, 
        typename std::enable_if<(Length * kElementSize % 16 != 0 || Start * kElementSize % 16 != 0) &&
                                (Length * kElementSize %  8 != 0 || Start * kElementSize %  8 != 0), int>::type Factor = (Length * kElementSize >> 2)>
    COGITO_DEVICE
    static void load(ShapedTensor<T, TensorSize>& tensor, const T* ptr) {
        static_assert(Start + Length <= TensorSize, "Load size exceed tensor size");
        COGITO_PRAGMA_UNROLL
        for (int i = 0; i < Factor; ++i) {
            ptx::ld_32b<kLdPolicy>(static_cast<void*>(reinterpret_cast<int8_t*>(&tensor[Start]) + i * 4), 
                static_cast<void*>(reinterpret_cast<int8_t*>(const_cast<T*>(ptr)) + i * 4));
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
        ThreadLd::load<TensorSize, Start + RangeStart * BlockSize, BlockSize>(tensor, ptr);
        ThreadLd::stripedLoad<Start, BlockSize, LineSize>(tensor, ptr + LineSize, mp::Range2Type<RangeStart + 1, RangeEnd>{});
    }
    template<int Start, int BlockSize, int LineSize, int TensorSize, int RangeEnd>
    COGITO_DEVICE
    static void stripedLoad(ShapedTensor<T, TensorSize>& tensor, const T* ptr, mp::Range2Type<RangeEnd, RangeEnd> /* unused */) {}

    template<int Start, int BlockSize, int TensorSize, int RangeStart, int RangeEnd>
    COGITO_DEVICE
    static void stripedLoad(ShapedTensor<T, TensorSize>& tensor, const T* ptr, const int ldg, mp::Range2Type<RangeStart, RangeEnd> /* unused */) {
        static_assert(Start + (RangeEnd - RangeStart - 1) * BlockSize < TensorSize);
        ThreadLd::load<TensorSize, Start + RangeStart * BlockSize, BlockSize>(tensor, ptr);
        ThreadLd::stripedLoad<Start, BlockSize>(tensor, ptr + ldg, ldg, mp::Range2Type<RangeStart + 1, RangeEnd>{});
    }
    template<int Start, int BlockSize, int TensorSize, int RangeEnd>
    COGITO_DEVICE
    static void stripedLoad(ShapedTensor<T, TensorSize>&, const T*, const int, mp::Range2Type<RangeEnd, RangeEnd> /* unused */) {}
};

///////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, StorePolicy StPolicy = StorePolicy::kDefault>
struct ThreadSt {
public:
    static constexpr int kElementSize = sizeof(T);
    static_assert(mp::IsPow2<kElementSize>::value, "Invalid element size");
    static constexpr StorePolicy kStPolicy = StPolicy;

public:
    // Store by STG/S.128
    template<int TensorSize, int Start = 0, int Length = TensorSize, 
        typename std::enable_if<(Start  * kElementSize % 16 == 0) && 
                                (Length * kElementSize % 16 == 0), int>::type Factor = (Length * kElementSize >> 4)>
    COGITO_DEVICE
    static void store(const ShapedTensor<T, TensorSize>& tensor, T* ptr) {
        static_assert(Start + Length <= TensorSize, "Load size exceed tensor size");
        COGITO_PRAGMA_UNROLL
        for (int i = 0; i < Factor; ++i) {
            ptx::st_128b<kStPolicy>(static_cast<void*>(reinterpret_cast<int8_t*>(ptr) + i * 16), 
                static_cast<void*>(reinterpret_cast<int8_t*>(const_cast<T*>(&tensor[Start])) + i * 16));
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
            ptx::st_64b<kStPolicy>(static_cast<void*>(reinterpret_cast<int8_t*>(ptr) + i * 8), 
                static_cast<void*>(reinterpret_cast<int8_t*>(const_cast<T*>(&tensor[Start])) + i * 8));
        }
    }

    template<int TensorSize, int Start = 0, int Length = TensorSize, 
        typename std::enable_if<(Length * kElementSize % 16 != 0 || Start * kElementSize % 16 != 0) &&
                                (Length * kElementSize %  8 != 0 || Start * kElementSize %  8 != 0), int>::type Factor = (Length * kElementSize >> 2)>
    COGITO_DEVICE
    static void store(const ShapedTensor<T, TensorSize>& tensor, T* ptr) {
        static_assert(Start + Length <= TensorSize, "Load size exceed tensor size");
        COGITO_PRAGMA_UNROLL
        for (int i = 0; i < Factor; ++i) {
            ptx::st_32b<kStPolicy>(static_cast<void*>(reinterpret_cast<int8_t*>(ptr) + i * 4), 
                static_cast<void*>(reinterpret_cast<int8_t*>(const_cast<T*>(&tensor[Start])) + i * 4));
        }
    }


    template<int Start, int BlockSize, int LineSize, int TensorSize, int RangeStart, int RangeEnd>
    COGITO_DEVICE
    static void stripedStore(const ShapedTensor<T, TensorSize>& tensor, T* ptr, mp::Range2Type<RangeStart, RangeEnd> /* unused */) {
        static_assert(Start + (RangeEnd - RangeStart - 1) * BlockSize < TensorSize);
        ThreadSt::store<TensorSize, Start + RangeStart * BlockSize, BlockSize>(tensor, ptr);
        ThreadSt::stripedStore<Start, BlockSize, LineSize>(tensor, ptr + LineSize, mp::Range2Type<RangeStart + 1, RangeEnd>{});
    }
    template<int Start, int BlockSize, int LineSize, int TensorSize, int RangeEnd>
    COGITO_DEVICE
    static void stripedStore(const ShapedTensor<T, TensorSize>&, T*, mp::Range2Type<RangeEnd, RangeEnd> /* unused */) {}

    template<int Start, int BlockSize, int TensorSize, int RangeStart, int RangeEnd>
    COGITO_DEVICE
    static void stripedStore(const ShapedTensor<T, TensorSize>& tensor, T* ptr, const int ldg, mp::Range2Type<RangeStart, RangeEnd> /* unused */) {
        static_assert(Start + (RangeEnd - RangeStart - 1) * BlockSize < TensorSize);
        ThreadSt::store<TensorSize, Start + RangeStart * BlockSize, BlockSize>(tensor, ptr);
        ThreadSt::stripedStore<Start, BlockSize>(tensor, ptr + ldg, ldg, mp::Range2Type<RangeStart + 1, RangeEnd>{});
    }
    template<int Start, int BlockSize, int TensorSize, int RangeEnd>
    COGITO_DEVICE
    static void stripedStore(const ShapedTensor<T, TensorSize>&, T*, const int, mp::Range2Type<RangeEnd, RangeEnd> /* unused */) {}
};

///////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, LoadPolicy LdPolicy = LoadPolicy::kDefault>
struct WarpLd {
public:
    static constexpr int kElementSize = sizeof(T);
    static_assert(mp::IsPow2<kElementSize>::value, "Invalid element size");
    static constexpr LoadPolicy  kLdPolicy = LdPolicy;

public:
    template<int Start, int BlockSize, int TensorSize>
    COGITO_DEVICE
    static void load(ShapedTensor<T, TensorSize>& tensor, const T* ptr, const uint32_t mask) {
        int laneid = ptx::getLaneid();
        if (((1u << laneid) & mask) != 0) {
            ThreadLd<T, kLdPolicy>::load<TensorSize, Start, BlockSize>(tensor, ptr + laneid * BlockSize);
        }
    }

    template<int Start, int BlockSize, int TensorSize>
    COGITO_DEVICE
    static void load(ShapedTensor<T, TensorSize>& tensor, const T* ptr) {
        int laneid = ptx::getLaneid();
        ThreadLd<T, kLdPolicy>::load<TensorSize, Start, BlockSize>(tensor, ptr + laneid * BlockSize);
    }
};

template<typename T, StorePolicy StPolicy = StorePolicy::kDefault>
struct WarpSt {
public:
    static constexpr int kElementSize = sizeof(T);
    static_assert(mp::IsPow2<kElementSize>::value, "Invalid element size");
    static constexpr StorePolicy kStPolicy = StPolicy;

public:
    template<int Start, int BlockSize, int TensorSize>
    COGITO_DEVICE
    static void store(const ShapedTensor<T, TensorSize>& tensor, T* ptr, const uint32_t mask) {
        int laneid = ptx::getLaneid();
        if (((1u << laneid) & mask) != 0) {
            ThreadSt<T, kStPolicy>::store<TensorSize, Start, BlockSize>(tensor, ptr + laneid * BlockSize);
        }
    }

    template<int Start, int BlockSize, int TensorSize>
    COGITO_DEVICE
    static void store(const ShapedTensor<T, TensorSize>& tensor, T* ptr) {
        int laneid = ptx::getLaneid();
        ThreadSt<T, kStPolicy>::store<TensorSize, Start, BlockSize>(tensor, ptr + laneid * BlockSize);
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, int GroupSize, LoadPolicy LdPolicy = LoadPolicy::kDefault>
struct ThreadGroupLd {
public:
    static constexpr int kGroupSize   = GroupSize;
    static constexpr int kElementSize = sizeof(T);
    static_assert(mp::IsPow2<kElementSize>::value, "Invalid element size");
    static constexpr LoadPolicy  kLdPolicy = LdPolicy;

public:
    template<int Start, int BlockSize, int TensorSize, int RangeStart, int RangeEnd>
    COGITO_DEVICE
    static void load(ShapedTensor<T, TensorSize>& tensor, const T* ptr, mp::Range2Type<RangeStart, RangeEnd> /* unused */) {
        int group_id = threadIdx.x % kGroupSize;
        ThreadLd<T, kLdPolicy>::stripedLoad<Start, BlockSize, BlockSize * kGroupSize>(tensor, ptr + group_id * BlockSize, mp::Range2Type<RangeStart, RangeEnd>{});
    }
};

template<typename T, int GroupSize, StorePolicy StPolicy = StorePolicy::kDefault>
struct ThreadGroupSt {
public:
    static constexpr int kGroupSize   = GroupSize;
    static constexpr int kElementSize = sizeof(T);
    static_assert(mp::IsPow2<kElementSize>::value, "Invalid element size");
    static constexpr StorePolicy kStPolicy = StPolicy;

public:
    template<int Start, int BlockSize, int TensorSize, int RangeStart, int RangeEnd>
    COGITO_DEVICE
    static void store(const ShapedTensor<T, TensorSize>& tensor, T* ptr, mp::Range2Type<RangeStart, RangeEnd> /* unused */) {
        int group_id = threadIdx.x % kGroupSize;
        ThreadSt<T, kStPolicy>::stripedStore<Start, BlockSize, BlockSize * kGroupSize>(tensor, ptr + group_id * BlockSize, mp::Range2Type<RangeStart, RangeEnd>{});
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cogito

