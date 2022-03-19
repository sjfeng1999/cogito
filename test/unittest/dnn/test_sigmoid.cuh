//
//
//
//

#pragma once 

#include "unittest/dnn/fixture_dnn.cuh"

///////////////////////////////////////////////////////////////////////////////////////////////

TEST_P(DNN2dFixture, SigmoidTest){
    
    status = cogito::dnn::Sigmoid<float>()(input_d, output_d, size);
    EXPECT_EQ(cudaSuccess, status);
    EXPECT_EQ(cudaSuccess, cudaMemcpy(output_h, 
                                      output_d, 
                                      size * sizeof(float), 
                                      cudaMemcpyDeviceToHost));

    Sigmoid<float> op;
    for (int i = 0; i < size; ++i){
        op(input_h + i, output_naive + i);
    }
    EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());
    EXPECT_TRUE(cogito::test::verifyResult<float>(output_h, output_naive, size));
};
