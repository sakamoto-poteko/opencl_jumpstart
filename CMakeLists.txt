cmake_minimum_required(VERSION 2.8)

project(offdpm)

find_package(OpenCL REQUIRED)

add_executable(${PROJECT_NAME} "main.cpp" "clutils.cpp")
add_definitions(-DCL_HPP_TARGET_OPENCL_VERSION=120)
add_definitions(-DCL_HPP_MINIMUM_OPENCL_VERSION=110)

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 11)
target_link_libraries(${PROJECT_NAME} OpenCL)
