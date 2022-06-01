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

template<LoadCachePolicy cache>
COGITO_DEVICE 
void ld_128b(void* dst, void* src);

// ldg / lds

template<LoadCachePolicy cache>
COGITO_DEVICE 
void ld_64b();

template<LoadCachePolicy cache>
COGITO_DEVICE 
void ld_32b();


template<LoadCachePolicy cache>
COGITO_DEVICE 
void ld_128b(void* dst, void* src);

template<LoadCachePolicy cache>
COGITO_DEVICE 
void ldg_64b();

template<LoadCachePolicy cache>
COGITO_DEVICE 
void ldg_32b();

// template<>
// COGITO_DEVICE 
// void ld_128b<LdCache::kCA>(void* dst, void* src) {
//     asm volatile(
//         "ld.global.v4.float     %0,     %1;\n\t"
//         :"=l"(dst) 
//         :"l"(src)
//         :"memory"
//     );
// }


} // namespace ptx
} // namespace cogito
