cmake_minimum_required(VERSION 3.10)
project(visual_odometry_tools)
message("   -- [PROJECT]: " visual_odometry_tools)

set(CMAKE_BUILD_TYPE Release) # Release, RelWithDebInfo
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O2 -pthread -pg")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}) 

find_package(OpenCV 4 REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(${CMAKE_SOURCE_DIR})
include_directories ("/usr/include/opencv4/")
include_directories(${Eigen3_INCLUDE_DIRS})

set(SRC_MOTION_ESTIMATOR ${SRC_MOTION_ESTIMATOR}
  motion_estimator/motion_estimator.cpp
)

set(SRC_UTIL ${SRC_UTIL}
  util/geometry_library.cpp
  util/pose3d.cpp
  util/timer.cpp
  util/triangulate_3d.cpp)

add_library(visual_odometry_tools STATIC ${SRC_MOTION_ESTIMATOR} ${SRC_UTIL})
target_link_libraries(visual_odometry_tools ${Eigen3_LIBRARIES} ${OpenCV_LIBS})
install(TARGETS visual_odometry_tools 
  DESTINATION ${PROJECT_NAME}
  CONFIGURATIONS Release)