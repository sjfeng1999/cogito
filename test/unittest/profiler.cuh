//
//
//
//

#pragma once

#include <stdio.h>
#include <chrono>
#include <limits>

#include "cogito/cogito.cuh"

namespace cogito {
namespace test {
    
///////////////////////////////////////////////////////////////////////////////////////////////

struct KernelProfiler {
    
    static constexpr int kWarmupTimes = 5;
    static constexpr int kRepeatTimes = 3;

private:
    float minVal;
    float maxVal;
    float avgVal;
    cudaError_t status;

public:
    template<typename KernelOp, typename... Args>
    Status profile(Args... args){
        
        KernelOp op;
        float elapsed = 0;

        minVal = std::numeric_limits<float>::max();
        maxVal = avgVal = 0;

        for (int i = 0; i < kWarmupTimes; ++i){
            op(args...);
        }

        for (int i = 0; i < kRepeatTimes; ++i){
            auto start_t = std::chrono::system_clock::now();
            status = op(args...);
            auto stop_t = std::chrono::system_clock::now();

            elapsed = std::chrono::duration_cast<std::chrono::microseconds>(stop_t - start_t).count();
            minVal = min(minVal, elapsed);
            maxVal = max(maxVal, elapsed);
            avgVal += elapsed;
        }
        avgVal /= kRepeatTimes;

        printf("    Elapsed > min = %5.2f   max = %5.2f   avg = %5.2f", minVal, maxVal, avgVal);
        return status == cudaSuccess ? Status::kSuccess : Status::kUnknownError;
    }
};


} // namespace test
} // namespace cogito