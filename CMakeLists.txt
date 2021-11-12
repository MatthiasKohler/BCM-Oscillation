cmake_minimum_required(VERSION 3.10)

project(rabit)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wno-undef -O3")

set(CMAKE_BUILD_TYPE Debug)

find_package(Eigen3 3.3.7 REQUIRED NO_MODULE)
include_directories(${EIGEN3_INCLUDE_DIR})
message(STATUS "EIGEN: " ${EIGEN3_VERSION_STRING} " (internal)")





set(Boost_USE_STATIC_LIBS OFF) 
set(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME OFF)
#find_package(Boost REQUIRED COMPONENTS GIL)
#include_directories(${Boost_INCLUDE_DIRS})
include_directories("/home/findus/info/Rabit/src/boost")
link_directories("/home/findus/info/Rabit/src/boost/stage/lib")

#include_directories("/home/findus/info/Rabit/src/pybind11/include")
#add_subdirectory("/home/findus/info/Rabit/src/pybind11")

#find_package(Boost REQUIRED COMPONENTS GIL)


find_package(HighFive REQUIRED)

add_executable(closed_loop_independent_pendulums closed_loop_independent_pendulums.cpp)
add_executable(closed_loop_double_pendulum closed_loop_double_pendulum.cpp)


add_library(neural_network neural_network.cpp)

target_link_libraries(closed_loop_independent_pendulums Eigen3::Eigen neural_network jpeg ${OpenCV_LIBS} HighFive stdc++fs)
target_link_libraries(closed_loop_double_pendulum Eigen3::Eigen neural_network jpeg ${OpenCV_LIBS} HighFive stdc++fs)


install(TARGETS closed_loop_independent_pendulums DESTINATION bin)
install(TARGETS closed_loop_double_pendulum DESTINATION bin)
