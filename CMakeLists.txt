cmake_minimum_required(VERSION 3.12.0)
project(main LANGUAGES CXX C)

include(./common/cmake-common.cmake)

include_directories(include)

FILE(GLOB SRC_FILES "src/*.cpp")
add_executable(main ${SRC_FILES})

deal_ii_setup_target(main)