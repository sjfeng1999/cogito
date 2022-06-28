//
//
//
//

#pragma once

#include "unittest/general/fixture_general.cuh"

TEST_P(GeneralFixture, ElementWiseUnaryTest){
    using TestElementWiseOpT = cogito::general::ElementWise<float, Square>;

    gflops = 1.0 * size / (1024 * 1024 * 1024);
    profiler.profile<TestElementWiseOpT>(gflops, input_d, output_d, size);
    TestElementWiseOpT()(input_d, output_d, size);
    EXPECT_EQ(cudaSuccess, cudaMemcpy(output_h, 
                                      output_d, 
                                      size * sizeof(float), 
                                      cudaMemcpyDeviceToHost));

    Square<float> op;
    for (int i = 0; i < size; ++i){
        output_naive[i] = op(input_h[i]);
    }
    EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());
    EXPECT_TRUE(cogito::test::verifyResult<float>(output_h, output_naive, size));
};

TEST_P(GeneralFixture, ElementWiseWithOperandTest){
    using TestElementWiseOpT = cogito::general::ElementWise<float, Sub>;
    float operand = 24;
    
    gflops = 1.0 * size / (1024 * 1024 * 1024);
    profiler.profile<TestElementWiseOpT, float*, float*, float, int>(gflops, input_d, output_d, operand, size);
    TestElementWiseOpT()(input_d, output_d, operand, size);
    EXPECT_EQ(cudaSuccess, cudaMemcpy(output_h, 
                                      output_d, 
                                      size * sizeof(float), 
                                      cudaMemcpyDeviceToHost));
    Sub<float> op;
    for (int i = 0; i < size; ++i){
        output_naive[i] = op(input_h[i], operand);
    }
    EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());
    EXPECT_TRUE(cogito::test::verifyResult<float>(output_h, output_naive, size));
};


TEST_P(GeneralFixture, ReduceWithIdentityTest){
    using TestReduceOpT = cogito::general::Reduce<float, Add>;
    
    gflops = 1.0 * size / (1024 * 1024 * 1024);
    profiler.profile<TestReduceOpT, float*, float*, int>(gflops, input_d, output_d, size);
    TestReduceOpT()(input_d, output_d, size);
    EXPECT_EQ(cudaSuccess, cudaMemcpy(output_h, 
                                      output_d, 
                                      sizeof(float), 
                                      cudaMemcpyDeviceToHost));
    Add<float> op;
    for (int i = 1; i < size; ++i){
        output_naive[0] = op(output_naive[0], output_naive[i]);
    }
    EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());
    EXPECT_TRUE(cogito::test::verifyResult<float>(output_h, output_naive, 1));
};
