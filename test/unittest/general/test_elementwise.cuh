//
//
//
//

#pragma once 

#include "unittest/general/test_general.cuh"

///////////////////////////////////////////////////////////////////////////////////////////////

TEST_P(GeneralFixture, ElementWiseTest){

    
    Square<float> op;
    for (int i = 0; i < size; ++i){
        op(input_h + i, output_naive + i);
        // printf("input: %.4f   output: %.4f", *(input_h + i), *(output_naive + i));
    }
    status = cogito::general::ElementWise<float, Square>()(input_d, output_d, size);

    EXPECT_EQ(cudaSuccess, status);
    EXPECT_EQ(cudaSuccess, cudaMemcpy(output_h, 
                                      output_d, 
                                      size * sizeof(float), 
                                      cudaMemcpyDeviceToHost));

    EXPECT_TRUE(cogito::test::verifyResult<float>(output_h, output_naive, size));
};

INSTANTIATE_TEST_SUITE_P(GeneralPart,
                         GeneralFixture,
                         testing::Values(32, 128, 2048));