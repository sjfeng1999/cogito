//
// 
//
//

#pragma once

namespace cogito {

///////////////////////////////////////////////////////////////////////////////////////////////

#define COGITO_GLOBAL               __global__
#define COGITO_DEVICE               __forceinline__  __device__ 
#define COGITO_HOST_DEVICE          __forceinline__  __host__ __device__ 

#define COGITO_PRAGMA_UNROLL        #pragma unroll
#define COGITO_LAUNCH_BOUND(n)      __launch__bounds__(n)

#define UPPER_DIV(x, y)             (((x) + (y) - 1) / y)

constexpr int kWarpSize = 32;

enum class Status {
    kSuccess,
};

} // namespace cogito
