
add_subdirectory(thirdparty/googletest)
set(TEST_SRC test.cu)

message(STATUS ">>>>>>>> COGITO TEST")
message(STATUS ${TEST_SRC})
message(STATUS "<<<<<<<<")


set(TARGET_ARCH "-gencode arch=compute_75,code=sm_75")
set(CMAKE_CUDA_FLAGS "${CUDA_NVCC_FLAGS} ${TARGET_ARCH}")

add_executable(cogito_test ${TEST_SRC})

target_include_directories(cogito_test 
  PUBLIC 
    ${PROJECT_SOURCE_DIR}/include 
    ${PROJECT_SOURCE_DIR}/test
    thirdparty/googletest/include
)

target_compile_features(cogito_test
  PRIVATE
    cxx_std_14
)

target_link_libraries(cogito_test 
  PRIVATE
    gtest 
    cublas
)

set_target_properties(cogito_test 
  PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    LINKER_LANGUAGE CUDA
)


