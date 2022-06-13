//
// CUDA ptx.cuh 
// export ptx-level instruction
//

#pragma once 

#include <tuple>
#include <type_traits>

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


template<int Scale, typename T, typename U = typename std::remove_const<T>::type>
COGITO_DEVICE
U* computeEffectiveAddr(T* ptr, int offset) {
    U* ptr_lea;
    asm volatile (
        "{                                 \n\t"
        ".reg.s64 offset;                  \n\t"
        "shl.b64 offset, %2, %3;           \n\t"
        "add.s64 %0, %1, offset;           \n\t"
        "}                                 \n\t"
        :"=l"(ptr_lea)
        :"l"(ptr), "l"(static_cast<long>(offset)), "r"(Scale)
        :"memory"
    );
    return ptr_lea;
}

///////////////////////////////////////////////////////////////////////////////////////////////

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

///////////////////////////////////////////////////////////////////////////////////////////////

#define cogito_load_128b(loadPolicy, loadModifier)                                               \
    template<>                                                                                   \
    COGITO_DEVICE                                                                                \
    void ld_128b<loadPolicy>(void* dst, void* src) {                                             \
        float4 val;                                                                              \
        asm volatile(                                                                            \
            "ld."#loadModifier".v4.f32     { %0, %1, %2, %3 },    [%4];  \n\t"                   \
            :"=f"(val.x), "=f"(val.y), "=f"(val.z), "=f"(val.w)                                  \
            :"l"(src)                                                                            \
            :"memory"                                                                            \
        );                                                                                       \
        *static_cast<float4*>(dst) = val;                                                        \
    }  

#define cogito_load_64b(loadPolicy, loadModifier)                                                \
    template<>                                                                                   \
    COGITO_DEVICE                                                                                \
    void ld_64b<loadPolicy>(void* dst, void* src) {                                              \
        float2 val;                                                                              \
        asm volatile(                                                                            \
            "ld."#loadModifier".v2.f32     { %0, %1 },    [%2];  \n\t"                           \
            :"=f"(val.x), "=f"(val.y)                                                            \
            :"l"(src)                                                                            \
            :"memory"                                                                            \
        );                                                                                       \
        *static_cast<float2*>(dst) = val;                                                        \
    }  

#define cogito_load_32b(loadPolicy, loadModifier)                                                \
    template<>                                                                                   \
    COGITO_DEVICE                                                                                \
    void ld_32b<loadPolicy>(void* dst, void* src) {                                              \
        float val;                                                                               \
        asm volatile(                                                                            \
            "ld."#loadModifier".f32     %0,    [%1];  \n\t"                                      \
            :"=f"(val)                                                                           \
            :"l"(src)                                                                            \
            :"memory"                                                                            \
        );                                                                                       \
        *static_cast<float*>(dst) = val;                                                         \
    }  

cogito_load_128b(LoadPolicy::kCA, global.ca)
cogito_load_128b(LoadPolicy::kCG, global.cg)
cogito_load_128b(LoadPolicy::kCS, global.cs)

cogito_load_64b(LoadPolicy::kCA, global.ca)
cogito_load_64b(LoadPolicy::kCG, global.cg)
cogito_load_64b(LoadPolicy::kCS, global.cs)

cogito_load_32b(LoadPolicy::kCA, global.ca)
cogito_load_32b(LoadPolicy::kCG, global.cg)
cogito_load_32b(LoadPolicy::kCS, global.cs)


#undef cogito_load_128b
#undef cogito_load_64b
#undef cogito_load_32b

///////////////////////////////////////////////////////////////////////////////////////////////

#define cogito_store_128b(storePolicy, storeModifier)                                            \
    template<>                                                                                   \
    COGITO_DEVICE                                                                                \
    void st_128b<storePolicy>(void* dst, void* src) {                                            \
        const float4 val = *static_cast<float4*>(src);                                           \
        asm volatile(                                                                            \
            "st."#storeModifier".v4.f32     [%4],   { %0, %1, %2, %3 };  \n\t"                   \
            ::"f"(val.x), "f"(val.y), "f"(val.z), "f"(val.w), "l"(dst)                           \
            :"memory"                                                                            \
        );                                                                                       \
    }  

#define cogito_store_64b(storePolicy, storeModifier)                                             \
    template<>                                                                                   \
    COGITO_DEVICE                                                                                \
    void st_64b<storePolicy>(void* dst, void* src) {                                             \
        const float2 val = *static_cast<float2*>(src);                                           \
        asm volatile(                                                                            \
            "st."#storeModifier".v2.f32     [%2],   { %0, %1 };  \n\t"                           \
            ::"f"(val.x), "f"(val.y), "l"(dst)                                                   \
            :"memory"                                                                            \
        );                                                                                       \
    }  

#define cogito_store_32b(storePolicy, storeModifier)                                             \
    template<>                                                                                   \
    COGITO_DEVICE                                                                                \
    void st_32b<storePolicy>(void* dst, void* src) {                                             \
        const float val = *static_cast<float*>(src);                                             \
        asm volatile(                                                                            \
            "st."#storeModifier".f32     [%1],  %0;  \n\t"                                       \
            ::"f"(val), "l"(dst)                                                                 \
            :"memory"                                                                            \
        );                                                                                       \
    }  

cogito_store_128b(StorePolicy::kWT, global.wt)
cogito_store_64b(StorePolicy::kWT, global.wt)
cogito_store_32b(StorePolicy::kWT, global.wt)


#undef cogito_store_128b
#undef cogito_store_64b
#undef cogito_store_32b

///////////////////////////////////////////////////////////////////////////////////////////////

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

///////////////////////////////////////////////////////////////////////////////////////////////

} // namespace ptx
} // namespace cogito
