


add_subdirectory(thirdparty/googletest)

file(GLOB_RECURSE TEST_SRC unittest/*.h unittest/*.cpp)


message(STATUS ">>>>>>>> UNIT TEST")
message(STATUS ${TEST_SRC})
message(STATUS "<<<<<<<<")

add_executable(unit_test ${TEST_SRC})

target_include_directories(unit_test PUBLIC thirdparty/googletest/include)
target_link_libraries(unit_test gtest)
target_link_libraries(unit_test cogito)

set_target_properties(unit_test PROPERTIES LINKER_LANGUAGE CUDA)