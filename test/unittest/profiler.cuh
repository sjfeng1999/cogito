//
//
//
//

#pragma once

#include <stdio.h>
#include <limits>

#include "cogito/cogito.cuh"

namespace cogito {
namespace test {

///////////////////////////////////////////////////////////////////////////////////////////////

struct KernelProfiler {
public:
    static constexpr int kWarmupTimes = 2;
    static constexpr int kRepeatTimes = 5;

private:
    float minVal = std::numeric_limits<float>::max();
    float maxVal = 0;
    float avgVal = 0;
    cudaEvent_t start, stop;
    cudaError_t status;

public:
    KernelProfiler() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    template<typename KernelOp, typename GFlopsOp, typename... Args>
    Status profile(Args... args){

        Status res = Status::kSuccess;

#ifdef COGITO_KERNEL_PROFILER 
        KernelOp kernel_op;
        GFlopsOp gflops_op;

        float elapsed = 0;
        float gflops = gflops_op(args...);

        for (int i = 0; i < kWarmupTimes; ++i){
            kernel_op(args...);
        }
        for (int i = 0; i < kRepeatTimes; ++i){
            cudaEventRecord(start);
            kernel_op(args...);
            cudaEventRecord(stop);
            status = cudaDeviceSynchronize();

            cudaEventElapsedTime(&elapsed, start, stop);
            minVal = min(minVal, elapsed);
            maxVal = max(maxVal, elapsed);
            avgVal += elapsed;
        }

        avgVal /= kRepeatTimes;
        res = (status == cudaSuccess) ? Status::kSuccess : Status::kUnknownError;

        if (res == Status::kSuccess){
            printf("    Elapsed >> min = %5.2f ms   max = %5.2f ms   avg = %5.2f ms     >> Glops = %5.2f GFlops/s\n", 
                minVal, maxVal, avgVal, gflops * 1000 / avgVal);
        } else {
            printf("    Some Error Occured. \n");
        }
#endif // COGITO_KERNEL_PROFILER 

        return res;
    }
};


} // namespace test
} // namespace cogito
