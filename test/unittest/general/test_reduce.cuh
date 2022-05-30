//
//
//
//

#pragma once

#include "unittest/general/fixture_general.cuh"

///////////////////////////////////////////////////////////////////////////////////////////////

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