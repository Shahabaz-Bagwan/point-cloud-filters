# min. version of cmake required for this project
cmake_minimum_required( VERSION 3.10)

# name of the project
project(pointCloudFilters)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(EIGEN3_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/eigen")
set(FLANN_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/flann/src/cpp")

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# collect all the .cxx needed for executable 
filE(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

# Where to find .h/.hpp files
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}" "${EIGEN3_INCLUDE_DIR}" "${FLANN_INCLUDE_DIR}")

# Executable target name and srouce
add_executable (${PROJECT_NAME} ${SOURCES})

# Link dependencies
target_link_libraries (${PROJECT_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/flann/build/lib/libflann.so)

# set target properties
set_target_properties(${PROJECT_NAME}
      PROPERTIES
          RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin/$<CONFIG>")
