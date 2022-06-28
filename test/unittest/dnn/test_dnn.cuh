//
//
//
//

#pragma once

#include "unittest/dnn/fixture_dnn.cuh"

TEST_P(DNN2dFixture, SigmoidTest){

    using TestSigmoid = cogito::dnn::Sigmoid<float>;
    TestSigmoid()(input_d, output_d, inner_size);

    EXPECT_EQ(cudaSuccess, cudaMemcpy(output_h, 
                                      output_d, 
                                      inner_size * sizeof(float), 
                                      cudaMemcpyDeviceToHost));

    gflops = 1.0 * total_size / (1024 * 1024 * 1024);
    profiler.profile<TestSigmoid>(gflops, input_d, output_d, inner_size);

    for (int i = 0; i < inner_size; ++i) {
        output_naive[i] = 1 / (1 + exp(-input_h[i]));
    }
    EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());
    EXPECT_TRUE(cogito::test::verifyResult<float>(output_h, output_naive, inner_size));
};

TEST_P(DNN2dFixture, PReluTest) {

    using TestPRelu = cogito::dnn::PRelu<float>;
    TestPRelu()(batch_size, inner_size, input_d, alpha_d, output_d);

    EXPECT_EQ(cudaSuccess, cudaMemcpy(output_h, 
                                      output_d, 
                                      total_size * sizeof(float), 
                                      cudaMemcpyDeviceToHost));

    gflops = 1.0 * total_size / (1024 * 1024 * 1024);
    profiler.profile<TestPRelu>(gflops, batch_size, inner_size, input_d, alpha_d, output_d);

    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < inner_size; ++j) {
            output_naive[i * inner_size + j] = 
                input_h[i * inner_size + j] > 0 ? input_h[i * inner_size + j] : alpha_h[i] * input_h[i * inner_size + j];
        }
    }
    EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());
    EXPECT_TRUE(cogito::test::verifyResult<float>(output_h, output_naive, total_size));
}

// TEST_P(DNN2dFixture, SoftmaxTest){
    
//     status = cogito::dnn::Softmax<float>()(input_d, output_d, size);
//     EXPECT_EQ(cudaSuccess, status);
//     EXPECT_EQ(cudaSuccess, cudaMemcpy(output_h, 
//                                       output_d, 
//                                       size * sizeof(float), 
//                                       cudaMemcpyDeviceToHost));

//     float maxVal = std::numeric_limits<float>::min();
//     float sum = 0;
//     for (int i = 0; i < size; ++i){
//         maxVal = max(maxVal, input_h[i]);
//     }
//     for (int i = 0; i < size; ++i){
//         output_naive[i] = exp(input_h[i] - maxVal);
//         sum += output_naive[i];
//     }
//     for (int i = 0; i < size; ++i){
//         output_naive[i] /= sum;
//     }

//     EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());
//     EXPECT_TRUE(cogito::test::verifyResult<float>(output_h, output_naive, size));
// };
