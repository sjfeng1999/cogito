
add_subdirectory(thirdparty/googletest)
set(TEST_SRC test.cu)

message(STATUS ">>>>>>>> COGITO TEST")
message(STATUS ${TEST_SRC})
message(STATUS "<<<<<<<<")

add_executable(cogito_test ${TEST_SRC})

target_include_directories(cogito_test PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(cogito_test PUBLIC ${CMAKE_SOURCE_DIR}/test)
target_include_directories(cogito_test PUBLIC thirdparty/googletest/include)

target_link_libraries(cogito_test gtest)

set_target_properties(cogito_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(cogito_test PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(cogito_test PROPERTIES LINKER_LANGUAGE CUDA)


