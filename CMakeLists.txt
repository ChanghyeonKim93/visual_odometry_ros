cmake_minimum_required(VERSION 3.5.1)
project(scale_mono_vo_ros)
set(CMAKE_BUILD_TYPE RelWithDebInfo) # Release, RelWithDebInfo
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O2 -pthread -pg")
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
  
  image_transport
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
  src/ros_wrapper/mononode.cpp

  src/core/scale_mono_vo.cpp
  src/core/scale_mono_vo_backend.cpp
  src/core/scale_mono_vo_trackimage.cpp

  src/core/camera.cpp
  src/core/frame.cpp
  src/core/landmark.cpp
  src/core/keyframes.cpp

  src/core/feature_extractor.cpp
  src/core/feature_tracker.cpp
  src/core/motion_estimator.cpp

  src/core/ba_solver/sparse_bundle_adjustment.cpp

  src/core/scale_estimator/scale_estimator.cpp
  
  src/core/image_processing.cpp
  src/core/mapping.cpp

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
add_executable(vo_node src/ros_wrapper/main.cpp ${SRC})
target_link_libraries (vo_node ${catkin_LIBRARIES} ${Eigen_LIBRARIES})

# executables
add_executable(backend_node src/ros_wrapper/main_backend_only.cpp src/ros_wrapper/backend_node.cpp ${SRC})
target_link_libraries (backend_node ${catkin_LIBRARIES} ${Eigen_LIBRARIES})

# executabls
add_executable(test_orb_matching_node src/ros_wrapper/main_orbmatching_test.cpp ${SRC})
target_link_libraries (test_orb_matching_node ${catkin_LIBRARIES} ${Eigen_LIBRARIES})



# Additional modules
add_executable(image_cropper 
  src/image_cropper/main_cropper.cpp
  src/image_cropper/image_cropper.cpp)
target_link_libraries (image_cropper ${catkin_LIBRARIES} ${Eigen_LIBRARIES})
