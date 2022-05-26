//
// 
//
//

#pragma once

///////////////////////////////////////////////////////////////////////////////////////////////

#define COGITO_GLOBAL               __global__
#define COGITO_DEVICE               __device__ __forceinline__  
#define COGITO_HOST_DEVICE          __host__ __device__ __forceinline__  

#define COGITO_PRAGMA_UNROLL        #pragma unroll
#define COGITO_LAUNCH_BOUND(n)      __launch__bounds__(n)

#define UPPER_DIV(x, y)             (((x) + (y) - 1) / (y))


#define cogito_device_ptr
#define cogito_host_ptr

///////////////////////////////////////////////////////////////////////////////////////////////

namespace cogito {

constexpr int kWarpSize = 32;

enum class Status {
    kSuccess,
    kUnknownError,
};

} // namespace cogito
