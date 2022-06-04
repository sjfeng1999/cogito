//
// ptx.h 
// export ptx-level instruction
//

#pragma once 

#include <cstdint>

#include "cogito/cogito.cuh"

namespace cogito {
namespace ptx {

//////////////////////////////////////////////////////////////////////////////////////////////

COGITO_DEVICE 
int32_t getClock(){
    int32_t clock;
    asm volatile(
        "mov.u32    %0,     %%clock; \n\t"
        :"=r"(clock)
        ::"memory"
    );
    return clock;
}

COGITO_DEVICE 
int32_t getSmid(){
    int32_t smid;
    asm volatile(
        "mov.u32    %0,     %%smid; \n\t"
        :"=r"(smid)
        ::"memory"
    );
    return smid;
}

COGITO_DEVICE 
int32_t getWarpid(){
    int32_t warpid;
    asm volatile(
        "mov.u32    %0,     %%warpid; \n\t"
        :"=r"(warpid)
        ::"memory"
    );
    return warpid;
}


COGITO_DEVICE 
int32_t getLaneid(){
    int32_t laneid;
    asm volatile(
        "mov.u32    %0,     %%laneid; \n\t"
        :"=r"(laneid)
        ::"memory"
    );
    return laneid;
}

COGITO_DEVICE 
void barSync(){
    asm volatile(
        "bar.sync   0; \n\t"
    );
}


template<LoadPolicy policy = LoadPolicy::kDefault>
COGITO_DEVICE 
void ld_128b(void* dst, void* src) {
    *static_cast<float4*>(dst) = *static_cast<float4*>(src);
}

template<LoadPolicy policy = LoadPolicy::kDefault>
COGITO_DEVICE 
void ld_64b(void* dst, void* src) {
    *static_cast<float2*>(dst) = *static_cast<float2*>(src);
}

template<LoadPolicy policy = LoadPolicy::kDefault>
COGITO_DEVICE 
void ld_32b(void* dst, void* src) {
    *static_cast<float*>(dst) = *static_cast<float*>(src);
}

template<StorePolicy policy = StorePolicy::kDefault>
COGITO_DEVICE 
void st_128b(void* dst, void* src) {
    *static_cast<float4*>(dst) = *static_cast<float4*>(src);
}

template<StorePolicy policy = StorePolicy::kDefault>
COGITO_DEVICE 
void st_64b(void* dst, void* src) {
    *static_cast<float2*>(dst) = *static_cast<float2*>(src);
}

template<StorePolicy policy = StorePolicy::kDefault>
COGITO_DEVICE 
void st_32b(void* dst, void* src) {
    *static_cast<float*>(dst) = *static_cast<float*>(src);
}



} // namespace ptx
} // namespace cogito
