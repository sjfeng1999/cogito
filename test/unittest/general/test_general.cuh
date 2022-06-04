//
//
//
//

#pragma once

#include "unittest/general/fixture_general.cuh"

struct ElementWiseGFlops {
    float operator()(float*, float*, int size) {
        return static_cast<float>(size) / (1024 * 1024 * 1024);
    }
};

TEST_P(GeneralFixture, ElementWiseUnaryTest){
    
    using TestElementWiseOpT = cogito::general::ElementWise<float, Square>;

    profiler.profile<TestElementWiseOpT, ElementWiseGFlops>(input_d, output_d, size);
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
    
    // profiler.profile<TestElementWiseOpT, float*, float*, float, int>(input_d, output_d, operand, size);
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


TEST_P(GeneralFixture, ReduceTensor){
    
    cogito::general::Reduce<float, Add>()(input_tensor, output_tensor);

    EXPECT_EQ(cudaSuccess, cudaMemcpy(output_h, 
                                      output_tensor.data(), 
                                      output_tensor.size() * decltype(output_tensor)::kElementSize, 
                                      cudaMemcpyDeviceToHost));

    Add<float> op;
    float res = input_h[0];
    for (int i = 1; i < size; ++i){
        res = op(input_h[i], res);
    }
    EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());
    EXPECT_TRUE(cogito::test::verifyResult<float>(output_h, &res, 1));
    
};

