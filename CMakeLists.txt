cmake_minimum_required(VERSION 3.12.0)
project(main LANGUAGES CXX C)

include(./common/cmake-common.cmake)

include_directories(include src)

SET(CMAKE_CXX_FLAGS  "-g -Wall -Wextra")

FILE(GLOB SRC_FILES "src/main.cpp" "src/DTR.cpp")
FILE(GLOB SRC_FILES_MF "src/main_mf.cpp")

add_executable(matrixbased ${SRC_FILES})
add_executable(matrixfree ${SRC_FILES_MF})

deal_ii_setup_target(matrixbased)
deal_ii_setup_target(matrixfree)