//
//
//
//

#pragma once 

#include "unittest/general/fixture_general.cuh"

///////////////////////////////////////////////////////////////////////////////////////////////

TEST_P(GeneralFixture, ElementWiseTest){
    
    status = cogito::general::ElementWise<float, Square>()(input_d, output_d, size);
    EXPECT_EQ(cudaSuccess, status);
    EXPECT_EQ(cudaSuccess, cudaMemcpy(output_h, 
                                      output_d, 
                                      size * sizeof(float), 
                                      cudaMemcpyDeviceToHost));

    Square<float> op;
    for (int i = 0; i < size; ++i){
        op(input_h + i, output_naive + i);
    }
    EXPECT_TRUE(cogito::test::verifyResult<float>(output_h, output_naive, size));
};
