//
// 
//
//

#pragma once

namespace cogito {

///////////////////////////////////////////////////////////////////////////////////////////////

#define COGITO_DEVICE           __forceinline__ __device__
#define COGITO_HOST_DEVICE      __forceinline__ __host__ __device__


#define UPPER_DIV(x, y)         (((x) + (y) - 1) / y)

    
enum class Status {
    kSuccess,
};

} // namespace cogito
