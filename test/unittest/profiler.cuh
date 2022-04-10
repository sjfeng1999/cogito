//
//
//
//

#pragma once

#include <stdio.h>
#include <chrono>

#include "cogito/cogito.cuh"

namespace cogito {
namespace test {
    
///////////////////////////////////////////////////////////////////////////////////////////////

struct KernelProfiler {
    
    static constexpr int kRepeatTimes = 10;

private:
    float minVal;
    float maxVal;
    float avgVal;

public:
    template<typename KernelOp, typename... Args>
    void profile(Args... args){
        
        minVal = maxVal = avgVal = 0;
        float elapsed = 0;

        for (int i = 0; i < kRepeatTimes; ++i){

            auto start_t = std::chrono::system_clock::now();
            KernelOp op;
            op(args...);
            auto stop_t = std::chrono::system_clock::now();

            elapsed = std::chrono::duration_cast<std::chrono::microseconds>(stop_t - start_t).count();
            minVal = min(minVal, elapsed);
            maxVal = max(maxVal, elapsed);
            avgVal += elapsed;
        }

        avgVal /= kRepeatTimes;

        printf("    Elapsed > min = %5.2f   max = %5.2f   avg = %5.2f", minVal, maxVal, avgVal);
    }
};


} // namespace test
} // namespace cogito