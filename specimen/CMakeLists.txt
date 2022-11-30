cmake_minimum_required(VERSION 3.1)
project(specimen)
find_package(OpenCV REQUIRED)
#define some macro as input file and output file
add_compile_options("-DINPUT_IMAGE=\"${CMAKE_CURRENT_SOURCE_DIR}/src.png\"")
add_compile_options("-DOUTPUT_P4P=\"${CMAKE_CURRENT_SOURCE_DIR}/../input.p4p\"")
add_executable(specimen specimen.cpp)
target_link_libraries( specimen ${OpenCV_LIBS} )
