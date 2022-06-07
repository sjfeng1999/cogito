//
//
//
//

#include "gtest/gtest.h"

#define COGITO_KERNEL_PROFILER 

#include "unittest/general/test_general.cuh"
#include "unittest/blas/test_blas.cuh"
#include "unittest/dnn/test_dnn.cuh"

int main(){
    testing::InitGoogleTest();
    return RUN_ALL_TESTS();
}
