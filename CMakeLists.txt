# min. version of cmake required for this project
cmake_minimum_required( VERSION 3.10)

# name of the project
project(pointCloudFilters)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/lib)

set(EIGEN3_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/eigen")
set(FLANN_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/flann/src/cpp")

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# collect all the .cxx needed for executable 
filE(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

add_subdirectory(thirdparty/flann)

# Create library
add_library(${PROJECT_NAME} SHARED ${SOURCES})
target_link_libraries (${PROJECT_NAME} ${CMAKE_SOURCE_DIR}/build/thirdparty/flann/lib/libflann.so )

# Where to find .h/.hpp files
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}" "${EIGEN3_INCLUDE_DIR}" "${FLANN_INCLUDE_DIR}")

set(EXAMPLE pointCloudFiltersExample)
# Executable target name and srouce
add_executable (${EXAMPLE} ${CMAKE_CURRENT_SOURCE_DIR}/Example/main.cpp)

# Link dependencies
target_link_libraries (${EXAMPLE} ${CMAKE_CURRENT_SOURCE_DIR}/lib/libpointCloudFilters.so)
