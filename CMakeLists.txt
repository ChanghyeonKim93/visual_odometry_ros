cmake_minimum_required(VERSION 3.5.1)
project(scale_mono_vo_ros)
set(CMAKE_BUILD_TYPE Debug) # Release, RelWithDebInfo
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -pthread -pg -O2")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}) 
find_package(catkin REQUIRED cmake_modules)
find_package(OpenCV REQUIRED)
find_package(Eigen REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  cv_bridge
  
  message_filters
  image_transport

  geometry_msgs
  sensor_msgs
)

catkin_package(
  DEPENDS Eigen
  CATKIN_DEPENDS roscpp
  INCLUDE_DIRS include
)

set(SRC ${SRC} 
  src/scale_mono_vo.cpp
  
  src/camera.cpp
  src/feature_extractor.cpp
  src/image_processing.cpp
  
  src/dataset_loader.cpp

  src/timer.cpp
  src/signal_handler_linux.cpp
)

set(includeDirs ${includeDirs} ${CMAKE_CURRENT_BINARY_DIR})
set(includeDirs ${includeDirs} include)

include_directories( ${includeDirs} )
include_directories( ${catkin_INCLUDE_DIRS} )
include_directories( ${Eigen_INCLUDE_DIRS} )

# executables
add_executable(vo_node src/main.cpp ${SRC})
target_link_libraries (vo_node ${catkin_LIBRARIES} ${Eigen_LIBRARIES})