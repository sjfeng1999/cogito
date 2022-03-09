cmake_minimum_required(VERSION 3.14)

project(cogito)

enable_language(CUDA)
set(CMAKE_CXX_STANDARD 11)

file(GLOB_RECURSE COGITO_SRC include/*.h)


message(STATUS ">>>>>>>> cogito")
message(STATUS ${COGITO_SRC})
message(STATUS "<<<<<<<<")

add_library(cogito SHARED ${COGITO_SRC})

set_target_properties(cogito PROPERTIES LINKER_LANGUAGE CUDA)

add_subdirectory(test)

