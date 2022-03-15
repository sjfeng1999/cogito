
add_subdirectory(thirdparty/googletest)
set(TEST_SRC test.cu)

message(STATUS ">>>>>>>> COGITO TEST")
message(STATUS ${TEST_SRC})
message(STATUS "<<<<<<<<")

add_executable(unit_test ${TEST_SRC})

target_include_directories(unit_test PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(unit_test PUBLIC ${CMAKE_SOURCE_DIR}/test)
target_include_directories(unit_test PUBLIC thirdparty/googletest/include)

target_link_libraries(unit_test gtest)

set_target_properties(unit_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(unit_test PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(unit_test PROPERTIES LINKER_LANGUAGE CUDA)


