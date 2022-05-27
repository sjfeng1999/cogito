//
//
//
//

#pragma once 

#include "unittest/general/fixture_general.cuh"

///////////////////////////////////////////////////////////////////////////////////////////////

TEST_P(GeneralFixture, ElementWiseUnaryTest){
    
    using TestElementWiseOpT = cogito::general::ElementWise<float, Square>;

    profiler.profile<TestElementWiseOpT, float*, float*, int>(input_d, output_d, size);
    TestElementWiseOpT()(input_d, output_d, size);
    EXPECT_EQ(cudaSuccess, cudaMemcpy(output_h, 
                                      output_d, 
                                      size * sizeof(float), 
                                      cudaMemcpyDeviceToHost));

    Square<float> op;
    for (int i = 0; i < size; ++i){
        op(input_h + i, output_naive + i);
    }
    EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());
    EXPECT_TRUE(cogito::test::verifyResult<float>(output_h, output_naive, size));
};


TEST_P(GeneralFixture, ElementWiseWithOperandTest){
    
    using TestElementWiseOpT = cogito::general::ElementWise<float, Sub>;
    float operand = 24;
    
    profiler.profile<TestElementWiseOpT, float*, float*, float, int>(input_d, output_d, operand, size);
    TestElementWiseOpT()(input_d, output_d, operand, size);
    EXPECT_EQ(cudaSuccess, cudaMemcpy(output_h, 
                                      output_d, 
                                      size * sizeof(float), 
                                      cudaMemcpyDeviceToHost));
    Sub<float> op;
    for (int i = 0; i < size; ++i){
        op(input_h + i, output_naive + i, operand);
    }
    EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());
    EXPECT_TRUE(cogito::test::verifyResult<float>(output_h, output_naive, size));
};


TEST_P(GeneralFixture, ElementWiseTensor){
    
    cogito::general::ElementWise<float, Square>()(input_tensor, output_tensor);

    EXPECT_EQ(cudaSuccess, cudaMemcpy(output_h, 
                                      output_tensor.data(), 
                                      output_tensor.size() * decltype(output_tensor)::kElementSize, 
                                      cudaMemcpyDeviceToHost));

    Square<float> op;
    for (int i = 0; i < size; ++i){
        op(input_h + i, output_naive + i);
    }
    EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());
    EXPECT_TRUE(cogito::test::verifyResult<float>(output_h, output_naive, size));
};