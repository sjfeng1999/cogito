//
// 
//  basic macro, enum class definition and meta programming utils
//

#pragma once

#include <cstdint>
#include <type_traits>

///////////////////////////////////////////////////////////////////////////////////////////////

#define COGITO_GLOBAL                       __global__
#define COGITO_KERNEL                       __global__
#define COGITO_DEVICE                       __device__ __forceinline__  
#define COGITO_DEVICE_INVOKE                __device__ __noinline__  
#define COGITO_HOST_DEVICE                  __host__ __device__ __forceinline__  

#define COGITO_PRAGMA_UNROLL                #pragma unroll
#define COGITO_PRAGMA_NO_UNROLL             #pragma unroll 1
#define COGITO_LAUNCH_BOUNDS(n)             __launch_bounds__(n)

#define UPPER_DIV(x, y)                     (((x) + (y) - 1) / (y))


#define cogito_host_val
#define cogito_host_ptr

#define cogito_device_ptr
#define cogito_device_reg

#define cogito_shared_mem
#define cogito_shared_ptr

#define cogito_const_mem
#define cogito_const_ptr

///////////////////////////////////////////////////////////////////////////////////////////////

namespace cogito {

constexpr int kWarpSize = 32;

enum class Status : uint8_t {
    kSuccess,
    kTensorShapeMismatch,
    kUnimplemented,
    kUnknownError,
};

enum class DataType : uint8_t {
    kFloat32,
    kTFloat32,  
    kFloat64,
    kFloat16, 
    kBFloat16,
    kInt32,
    kInt8,
};

enum class AlignType : uint8_t {
    k128B,
    k64B,
    k32B,
};

enum class LoadPolicy : uint8_t {
    kCA,          // cache at all level
    kCG,          // cache at global-level
    kCS,          // cache streaming
    kLU,          // last use
    kCV,          // don't cache and fetch again
    kShared,      // load from shared memory
    kConstant,    // load from constant memory
    kDefault,     // default global memory cache modifier
};

enum class StorePolicy : uint8_t {
    kWB,          // cache write-back
    kCG,          // cache at global-level
    kCS,          // cache streaming
    kWT,          // cache write-through
    kShared,      // store into shared memory
    kDefault,     // default global memory cache modifier
};


template<AlignType type>
struct Chunk;


///////////////////////////////////////////////////////////////////////////////////////////////

namespace mp {

template<int Val>
struct Int2Type {
    static constexpr int value = Val;
};


template<int Start, int End>
struct Range2Type {
    static constexpr int start = Start;
    static constexpr int end   = End;
};


template<int Val, int... Vals>
struct Sum {
    static constexpr int value = Val + Sum<Vals...>::value;
};
template<int Val>
struct Sum<Val> {
    static constexpr int value = Val;
};


template<int Val, int... Vals>
struct Product {
    static constexpr int value = Val * Product<Vals...>::value;
};
template<int Val>
struct Product<Val> {
    static constexpr int value = Val;
};


template<int Val, int... Vals>
struct Back {
    static constexpr int value = Back<Vals...>::value;
};
template<int Val>
struct Back<Val> {
    static constexpr int value = Val;
};


template<int Val>
struct IsPow2 {
    static constexpr bool value = (Val & Val - 1) == 0;
};


template<int Val>
struct Pow2 {
    static constexpr int value = Pow2<(Val - 1)>::value << 1;
};
template<>
struct Pow2<0> {
    static constexpr int value = 1;
};


template<int Val>
struct Log2 {
    static_assert(IsPow2<Val>::value);
    static constexpr int value = 1 + Log2<(Val >> 1)>::value;
};
template<>
struct Log2<1> {
    static constexpr int value = 0;
};


template<int ThreadCount>
struct WarpMask {
    static constexpr uint value = (1 << (ThreadCount - 1)) | WarpMask<ThreadCount - 1>::value;
};
template<>
struct WarpMask<0> {
    static constexpr uint value = 0;
};

} // namesapce mp
} // namespace cogito
