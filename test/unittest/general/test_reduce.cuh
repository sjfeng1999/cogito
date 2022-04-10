//
//
//
//

#pragma once

#include "unittest/general/fixture_general.cuh"

///////////////////////////////////////////////////////////////////////////////////////////////

TEST_P(GeneralFixture, ReduceTest){

    using TestReduceOpT = cogito::general::Reduce<float, Add>;

    profiler.profile<TestReduceOpT, float*, float*, int>(input_d, output_d, size);
    TestReduceOpT()(input_d, output_d, size);
    EXPECT_EQ(cudaSuccess, cudaMemcpy(output_h, 
                                      output_d, 
                                      sizeof(float), 
                                      cudaMemcpyDeviceToHost));

    Add<float> op;
    float res = input_h[0];
    for (int i = 1; i < size; ++i){
        res = op(input_h + i, &res);
    }
    EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());
    EXPECT_TRUE(cogito::test::verifyResult<float>(output_h, &res, 1));
    
};