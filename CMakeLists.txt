# min. version of cmake required for this project
CMAKE_MINIMUM_REQUIRED( VERSION 3.18)

# name of the project
PROJECT(pointCloudFilters)

find_package( Eigen3 3.3 REQUIRED NO_MODULE)

set(FLANN_INCLUDE_DIR "/usr/local/include/flann")
# set(EIGEN3_INCLUDE_DIR "/usr/include/eigen3")

# specify the C++ standard
SET(CMAKE_CXX_STANDARD 14)
SET(CMAKE_CXX_STANDARD_REQUIRED True)

# collect all the .cxx needed for executable 
FILE(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

# Where to find .h/.hpp files
INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}/include" "${EIGEN3_INCLUDE_DIR}" "${FLANN_INCLUDE_DIR}")

# Executable target name and srouce
ADD_EXECUTABLE (${PROJECT_NAME} ${SOURCES})

# Link dependencies
TARGET_LINK_LIBRARIES (${PROJECT_NAME} Eigen3::Eigen /usr/local/lib/libflann_cpp.so)

# Provide other include dirs required for the project
# TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} ${HEADERS} )

# When installing where to install 
INSTALL(TARGETS ${PROJECT_NAME} DESTINATION ~/.local/bin) 
INSTALL(FILES ${cmake_current_source_dir}/include DESTINATION ~/.local/include) 