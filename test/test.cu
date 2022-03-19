//
//
//
//

#include "gtest/gtest.h"

#include "unittest/blas/test_blas.cuh"
#include "unittest/dnn/test_dnn.cuh"
#include "unittest/general/test_general.cuh"

int main(){
    testing::InitGoogleTest();
    return RUN_ALL_TESTS();
}