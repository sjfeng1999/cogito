//
//
//
//

#pragma once 

#include "unittest/dnn/fixture_dnn.cuh"

///////////////////////////////////////////////////////////////////////////////////////////////

TEST_P(DNN2dFixture, SoftmaxTest){
    
    status = cogito::dnn::Softmax<float>()(input_d, output_d, size);
    EXPECT_EQ(cudaSuccess, status);
    EXPECT_EQ(cudaSuccess, cudaMemcpy(output_h, 
                                      output_d, 
                                      size * sizeof(float), 
                                      cudaMemcpyDeviceToHost));

    float maxVal = std::numeric_limits<float>::min();
    float sum = 0;
    for (int i = 0; i < size; ++i){
        maxVal = max(maxVal, input_h[i]);
    }
    for (int i = 0; i < size; ++i){
        output_naive[i] = exp(input_h[i] - maxVal);
        sum += output_naive[i];
    }
    for (int i = 0; i < size; ++i){
        output_naive[i] /= sum;
    }

    EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());
    EXPECT_TRUE(cogito::test::verifyResult<float>(output_h, output_naive, size));
};
