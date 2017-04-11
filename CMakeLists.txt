cmake_minimum_required(VERSION 3.0)
project(stdmMf)

option(COMPILE_WITH_WARNINGS "Add -Wall and -Wextra" OFF)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake-modules)

add_subdirectory(libs/googletest)
add_subdirectory(libs/njm_cpp)

add_subdirectory(src/main)
add_subdirectory(src/bin)

include(CTest)
add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND})
add_subdirectory(test EXCLUDE_FROM_ALL)
