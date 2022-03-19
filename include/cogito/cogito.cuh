//
// 
//
//

#pragma once

namespace cogito {

///////////////////////////////////////////////////////////////////////////////////////////////

#define COGITO_GLOBAL           __global__
#define COGITO_DEVICE           __forceinline__  __device__ 
#define COGITO_HOST_DEVICE      __forceinline__  __host__ __device__ 

#define COGITO_UNROLL           #pragma unroll

#define UPPER_DIV(x, y)         (((x) + (y) - 1) / y)

constexpr int kWarpSize = 32;

enum class Status {
    kSuccess,
};

} // namespace cogito
