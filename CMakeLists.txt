cmake_minimum_required(VERSION 3.1)
project(color_icp)

set (CMAKE_CXX_STANDARD 14)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'RelWithDebInfo' as none was specified")
    set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING
            "Default build type: RelWithDebInfo" FORCE)
endif ()
message(STATUS "Current build type: ${CMAKE_BUILD_TYPE}")

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)

include_directories(
    include
    ${EIGEN3_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
)

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})


add_executable (color_icp src/color_icp.cpp)
target_link_libraries (color_icp ${PCL_LIBRARIES})

add_executable (optimization src/optimization.cpp)
target_link_libraries (optimization ${PCL_LIBRARIES})
