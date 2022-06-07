//
// 
//  basic macro, enum class definition and meta programming utils
//

#pragma once

#include <cstdint>

///////////////////////////////////////////////////////////////////////////////////////////////

#define COGITO_GLOBAL                       __global__
#define COGITO_KERNEL                       __global__
#define COGITO_DEVICE                       __device__ __forceinline__  
#define COGITO_HOST_DEVICE                  __host__ __device__ __forceinline__  

#define COGITO_PRAGMA_UNROLL                #pragma unroll
#define COGITO_PRAGMA_NO_UNROLL             #pragma unroll 1
#define COGITO_LAUNCH_BOUND(n)              __launch__bounds__(n)

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
    kUnknownError,
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
struct Product {
    static constexpr int value = Val * Product<Vals...>::value;
};

template<int Val>
struct Product<Val> {
    static constexpr int value = Val;
};

template<int Val>
struct IsPow2 {
    static constexpr int value = (Val & Val - 1) == 0;
};


} // namesapce mp
} // namespace cogito
