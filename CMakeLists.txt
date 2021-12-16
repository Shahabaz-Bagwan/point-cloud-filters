# min. version of cmake required for this project
cmake_minimum_required( VERSION 3.10)

# name of the project
project(pointCloudFilters)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/build/bin)
set(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/build/lib)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# collect all the .cxx needed for executable 
filE(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

add_subdirectory(thirdparty/flann)
add_subdirectory(example)

# Create library
add_library(${PROJECT_NAME}_shared SHARED ${SOURCES})

target_link_libraries(${PROJECT_NAME}_shared PUBLIC flann_cpp)

target_include_directories( ${PROJECT_NAME}_shared PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/flann/src/cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/eigen
)

add_library(${PROJECT_NAME}_static STATIC ${SOURCES})

target_link_libraries(${PROJECT_NAME}_static PUBLIC flann_cpp_s )

target_include_directories( ${PROJECT_NAME}_static PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/flann/src/cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/eigen
)