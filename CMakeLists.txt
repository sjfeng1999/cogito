cmake_minimum_required(VERSION 3.14)

project(cogito
        LANGUAGES
        CUDA CXX)

enable_language(CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED)

file(GLOB_RECURSE COGITO_SRC include/*.cuh)


message(STATUS ">>>>>>>> cogito")
message(STATUS ${COGITO_SRC})
message(STATUS "<<<<<<<<")

set(CMAKE_CUDA_FLAGS "${CUDA_NVCC_FLAGS} ${CUDA_OPT_FLAG} -Xcompiler -fPIC --compiler-options -fno-strict-aliasing \
        -lineinfo -Xptxas -dlcm=cg -use_fast_math -D_GLIBCXX_USE_CXX11_ABI=1")


add_library(cogito SHARED ${COGITO_SRC})
target_include_directories(cogito PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
set_target_properties(cogito PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(cogito PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(cogito PROPERTIES CUDA_STANDARD 11)
set_target_properties(cogito PROPERTIES LINKER_LANGUAGE CUDA)

add_subdirectory(test)

