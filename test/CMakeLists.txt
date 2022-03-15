
add_subdirectory(thirdparty/googletest)

enable_language(CUDA)

file(GLOB_RECURSE TEST_SRC test.cu unittest/*.cuh)

message(STATUS ">>>>>>>> UNIT TEST")
message(STATUS ${TEST_SRC})
message(STATUS "<<<<<<<<")

set(CMAKE_CUDA_FLAGS "${CUDA_NVCC_FLAGS} ${CUDA_OPT_FLAG} -Xcompiler -fPIC --compiler-options -fno-strict-aliasing \
        -lineinfo -Xptxas -dlcm=cg -use_fast_math -D_GLIBCXX_USE_CXX11_ABI=1")

add_executable(unit_test ${TEST_SRC})

target_include_directories(unit_test PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(unit_test PUBLIC ${CMAKE_SOURCE_DIR}/test)
target_include_directories(unit_test PUBLIC thirdparty/googletest/include)
target_include_directories(unit_test PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

set_target_properties(unit_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(unit_test PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(unit_test PROPERTIES CUDA_STANDARD 11)
set_target_properties(unit_test PROPERTIES LINKER_LANGUAGE CUDA)

target_link_libraries(unit_test cogito)
target_link_libraries(unit_test gtest)

