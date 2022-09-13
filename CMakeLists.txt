cmake_minimum_required(VERSION 3.5.1)
project(scale_mono_vo_ros)
set(CMAKE_BUILD_TYPE Release) # Release, RelWithDebInfo
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -pthread -pg -O2")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}) 
find_package(catkin REQUIRED cmake_modules)
find_package(OpenCV REQUIRED)
find_package(Eigen REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  cv_bridge
  
  message_generation

  geometry_msgs
  sensor_msgs
  nav_msgs
)

#messages
add_message_files(
  FILES
  statisticsStamped.msg 
)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs
)


catkin_package(
  DEPENDS Eigen
  CATKIN_DEPENDS roscpp message_runtime
  INCLUDE_DIRS include
)

set(SRC ${SRC} 
  src/mononode.cpp

  src/core/scale_mono_vo.cpp
  src/core/scale_mono_vo_localbundle.cpp
  src/core/scale_mono_vo_ap3p.cpp
  src/core/scale_mono_vo_naive_threeviews.cpp
  src/core/scale_mono_vo_backend.cpp
  src/core/scale_mono_vo_localbundle2.cpp

  src/core/camera.cpp
  src/core/frame.cpp
  src/core/landmark.cpp
  src/core/keyframes.cpp

  src/core/feature_extractor.cpp
  src/core/feature_tracker.cpp
  src/core/motion_estimator.cpp

  src/core/ba_solver/sparse_bundle_adjustment.cpp

  src/core/scale_estimator.cpp
  src/core/image_processing.cpp
  src/core/mapping.cpp

  src/core/dataset_loader.cpp

  src/util/timer.cpp
  src/util/signal_handler_linux.cpp
  src/util/geometry_library.cpp
  src/util/histogram.cpp
  src/util/trigonometry_fast.cpp
)

set(includeDirs ${includeDirs} ${CMAKE_CURRENT_BINARY_DIR})
set(includeDirs ${includeDirs} include)

include_directories( ${includeDirs} )
include_directories( ${catkin_INCLUDE_DIRS} )
include_directories( ${Eigen_INCLUDE_DIRS} )

# executables
add_executable(vo_node src/main.cpp ${SRC})
target_link_libraries (vo_node ${catkin_LIBRARIES} ${Eigen_LIBRARIES})

# executables
add_executable(backend_node src/main_backend_only.cpp src/backend_node.cpp ${SRC})
target_link_libraries (backend_node ${catkin_LIBRARIES} ${Eigen_LIBRARIES})

# executabls
add_executable(test_orb_matching_node src/main_orbmatching_test.cpp ${SRC})
target_link_libraries (test_orb_matching_node ${catkin_LIBRARIES} ${Eigen_LIBRARIES})
