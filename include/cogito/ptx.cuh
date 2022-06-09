//
// ptx.h 
// export ptx-level instruction
//

#pragma once 

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


template<>
COGITO_DEVICE 
void ld_128b<LoadPolicy::kCA>(void* dst, void* src) {
    float4 val;
    asm volatile(
#if __CUDA_ARCH__ >= 800
        "ld.global.ca.L2::256B.v4.f32     { %0, %1, %2, %3 },    [%4];  \n\t"
#else 
        "ld.global.ca.L2::128B.v4.f32     { %0, %1, %2, %3 },    [%4];  \n\t"
#endif
        :"=f"(val.x), "=f"(val.y), "=f"(val.z), "=f"(val.w)
        :"l"(src)
        :"memory"
    );
    *static_cast<float4*>(dst) = val;
}

template<>
COGITO_DEVICE 
void ld_64b<LoadPolicy::kCA>(void* dst, void* src) {
    float2 val;
    asm volatile(
#if __CUDA_ARCH__ >= 800
        "ld.global.ca.L2::256B.v2.f32     { %0, %1 },    [%2];  \n\t"
#else 
        "ld.global.ca.L2::128B.v2.f32     { %0, %1 },    [%2];  \n\t"
#endif
        :"=f"(val.x), "=f"(val.y)
        :"l"(src)
        :"memory"
    );
    *static_cast<float2*>(dst) = val;
}

template<>
COGITO_DEVICE 
void ld_32b<LoadPolicy::kCA>(void* dst, void* src) {
    float val;
    asm volatile(
#if __CUDA_ARCH__ >= 800
        "ld.global.ca.L2::256B.f32     %0,    [%1];  \n\t"
#else 
        "ld.global.ca.L2::128B.f32     %0,    [%1];  \n\t"
#endif
        :"=f"(val)
        :"l"(src)
        :"memory"
    );
    *static_cast<float*>(dst) = val;
}



template<>
COGITO_DEVICE 
void ld_128b<LoadPolicy::kCS>(void* dst, void* src) {
    float4 val;
    asm volatile(
#if __CUDA_ARCH__ >= 800
        "ld.global.cs.L2::256B.v4.f32     { %0, %1, %2, %3 },    [%4];  \n\t"
#else 
        "ld.global.cs.L2::128B.v4.f32     { %0, %1, %2, %3 },    [%4];  \n\t"
#endif
        :"=f"(val.x), "=f"(val.y), "=f"(val.z), "=f"(val.w)
        :"l"(src)
        :"memory"
    );
    *static_cast<float4*>(dst) = val;
}

template<>
COGITO_DEVICE 
void ld_64b<LoadPolicy::kCS>(void* dst, void* src) {
    float2 val;
    asm volatile(
#if __CUDA_ARCH__ >= 800
        "ld.global.cs.L2::256B.v2.f32     { %0, %1 },    [%2];  \n\t"
#else 
        "ld.global.cs.L2::128B.v2.f32     { %0, %1 },    [%2];  \n\t"
#endif
        :"=f"(val.x), "=f"(val.y)
        :"l"(src)
        :"memory"
    );
    *static_cast<float2*>(dst) = val;
}

template<>
COGITO_DEVICE 
void ld_32b<LoadPolicy::kCS>(void* dst, void* src) {
    float val;
    asm volatile(
#if __CUDA_ARCH__ >= 800
        "ld.global.cs.L2::256B.f32     %0,    [%1];  \n\t"
#else 
        "ld.global.cs.L2::128B.f32     %0,    [%1];  \n\t"
#endif
        :"=f"(val)
        :"l"(src)
        :"memory"
    );
    *static_cast<float*>(dst) = val;
}



template<>
COGITO_DEVICE 
void ld_128b<LoadPolicy::kShared>(void* dst, void* src) {
    float4 val;
    asm volatile(
        "ld.shared.v4.f32     { %0, %1, %2, %3 },    [%4];  \n\t"
        :"=f"(val.x), "=f"(val.y), "=f"(val.z), "=f"(val.w)
        :"l"(src)
        :"memory"
    );
    *static_cast<float4*>(dst) = val;
}



COGITO_DEVICE 
void prefetch_l2(void* src) {
    asm volatile (
        "prefetch.global.L2     [%0];\n\t"
        ::"l"(src)
        :"memory"
    );
}

COGITO_DEVICE 
void prefetch_l1(void* src) {
    asm volatile (
        "prefetch.global.L1     [%0];\n\t"
        ::"l"(src)
        :"memory"
    );
}


} // namespace ptx
} // namespace cogito
