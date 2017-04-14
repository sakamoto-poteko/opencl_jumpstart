cmake_minimum_required(VERSION 2.8)

project(offdpm)


add_executable(${PROJECT_NAME} "main.cpp" "clutils.cpp" "lodepng.cpp" "cltest.cl" "cl_helper.hpp")

find_package(OpenCL 1.2 REQUIRED)
include_directories(${OPENCL_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} OpenCL)

add_definitions(-DCL_HPP_TARGET_OPENCL_VERSION=120)
add_definitions(-DCL_HPP_MINIMUM_OPENCL_VERSION=120)

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 11)
