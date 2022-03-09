

message(STATUS ">>>>>>>> UNIT TEST")

add_subdirectory(thirdparty/googletest)

file(GLOB_RECURSE TEST_SRC unittest/*.h unittest/*.cpp)


add_executable(unittest ${TEST_SRC})

target_include_directories(unittest PUBLIC thirdparty/googletest/include)
target_link_libraries(unittest gtest)
target_link_libraries(unittest cogito)

set_target_properties(unittest PROPERTIES LINKER_LANGUAGE CUDA)