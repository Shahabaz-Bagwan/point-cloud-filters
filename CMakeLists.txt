# min. version of cmake required for this project
CMAKE_MINIMUM_REQUIRED( VERSION 3.10)

# name of the project
PROJECT(pclExtractedStuff)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(EIGEN3_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/eigen")
set(FLANN_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/flann/src/cpp")

# specify the C++ standard
SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_STANDARD_REQUIRED True)

# collect all the .cxx needed for executable 
FILE(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

# Where to find .h/.hpp files
INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}" "${EIGEN3_INCLUDE_DIR}" "${FLANN_INCLUDE_DIR}")

# Executable target name and srouce
ADD_EXECUTABLE (${PROJECT_NAME} ${SOURCES})

# Link dependencies
TARGET_LINK_LIBRARIES (${PROJECT_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/flann/build/lib/libflann.so)
