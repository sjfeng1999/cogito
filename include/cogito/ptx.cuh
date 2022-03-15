//
// ptx.h 
// export ptx-level instruction
//

#pragma once 

#include <cstdint>
#include "cuda.h"
#include "cuda_runtime.h"

#include "cogito/cogito.cuh"

namespace cogito {
namespace utils {

//////////////////////////////////////////////////////////////////////////////////////////////


COGITO_DEVICE 
int32_t get_clock(){
    int32_t clock;
    asm volatile(
        "mov.u32    %0,     %%clock; \n\t"
        :"=r"(clock)::"memory"
    );
    return clock;
}

COGITO_DEVICE 
int32_t get_smid(){
    int32_t smid;
    asm volatile(
        "mov.u32    %0,     %%smid; \n\t"
        :"=r"(smid)::"memory"
    );
    return smid;
}

COGITO_DEVICE 
int32_t get_warpid(){
    int32_t warpid;
    asm volatile(
        "mov.u32    %0,     %%warpid; \n\t"
        :"=r"(warpid)::"memory"
    );
    return warpid;
}

COGITO_DEVICE 
int32_t get_global_warpid(){
    int32_t global_warpid;
    int32_t local_warpid = get_warpid();
    int32_t block_id = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    int32_t warp_per_block = UPPER_DIV(blockDim.x * blockDim.y * blockDim.z, kWarpSize);
    global_warpid = block_id * warp_per_block + local_warpid;
    return global_warpid;
}


COGITO_DEVICE 
int32_t get_laneid(){
    int32_t laneid;
    asm volatile(
        "mov.u32    %0,     %%laneid; \n\t"
        :"=r"(laneid)::"memory"
    );
    return laneid;
}

COGITO_DEVICE 
void bar_sync(){
    asm volatile(
        "bar.sync   0; \n\t"
    );
}

} // namespace utils
} // namespace cogito