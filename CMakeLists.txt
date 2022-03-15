cmake_minimum_required(VERSION 3.14)

project(cogito
        LANGUAGES
        CUDA CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED)

add_subdirectory(test)
