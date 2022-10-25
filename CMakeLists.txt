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


set(SRC_MONO ${SRC_MONO} 
  src/ros_wrapper/mononode.cpp
  src/ros_wrapper/ros_print_in_color.cpp

  src/core/feature_extractor.cpp
  src/core/feature_tracker.cpp
  src/core/motion_estimator.cpp

  src/core/ba_solver/sparse_bundle_adjustment.cpp
  src/core/ba_solver/sparse_bundle_adjustment_scale_sqp.cpp

  src/core/camera.cpp
  src/core/frame.cpp
  src/core/landmark.cpp
  src/core/keyframes.cpp

  src/core/image_processing.cpp
  src/core/mapping.cpp

  src/util/timer.cpp
  src/util/stop_watch_timer.cpp
  src/util/signal_handler_linux.cpp
  src/util/geometry_library.cpp
  src/util/histogram.cpp
  src/util/trigonometry_fast.cpp

  src/core/scale_estimator/scale_estimator.cpp
  src/core/scale_estimator/scale_forward_propagation.cpp
  src/core/scale_estimator/absolute_scale_recovery.cpp
  
  src/core/scale_mono_vo/scale_mono_vo.cpp
  src/core/scale_mono_vo/scale_mono_vo_backend.cpp
  src/core/scale_mono_vo/scale_mono_vo_trackimage.cpp

  src/core/stereo_vo/stereo_vo.cpp
)


set(SRC_STEREO ${SRC_STEREO} 
  src/ros_wrapper/stereonode.cpp
  src/ros_wrapper/ros_print_in_color.cpp

  src/core/feature_extractor.cpp
  src/core/feature_tracker.cpp
  src/core/motion_estimator.cpp

  src/core/ba_solver/sparse_bundle_adjustment.cpp

  src/core/camera.cpp
  src/core/frame.cpp
  src/core/landmark.cpp
  src/core/keyframes.cpp

  src/core/image_processing.cpp
  src/core/mapping.cpp

  src/util/timer.cpp
  src/util/stop_watch_timer.cpp
  src/util/signal_handler_linux.cpp
  src/util/geometry_library.cpp
  src/util/histogram.cpp
  src/util/trigonometry_fast.cpp

  src/core/stereo_vo/stereo_vo.cpp
  src/core/stereo_vo/stereo_vo_trackimage.cpp
)

set(includeDirs ${includeDirs} ${CMAKE_CURRENT_BINARY_DIR})
set(includeDirs ${includeDirs} include)

include_directories( ${includeDirs} )
include_directories( ${catkin_INCLUDE_DIRS} )
include_directories( ${Eigen_INCLUDE_DIRS} )

# executable for 'scale_mono_vo' node
add_executable(vo_node src/ros_wrapper/main_scale_mono_vo.cpp ${SRC_MONO})
target_link_libraries (vo_node ${catkin_LIBRARIES} ${Eigen_LIBRARIES})

# executable for 'backend only' node
add_executable(backend_node src/ros_wrapper/main_backend_only.cpp src/ros_wrapper/backend_node.cpp ${SRC_MONO})
target_link_libraries (backend_node ${catkin_LIBRARIES} ${Eigen_LIBRARIES})

# executable for 'test orb matching' node
add_executable(test_orb_matching_node src/ros_wrapper/main_orbmatching_test.cpp ${SRC_MONO})
target_link_libraries (test_orb_matching_node ${catkin_LIBRARIES} ${Eigen_LIBRARIES})


# executable for 'stereo_vo' node
add_executable(stereo_vo_node src/ros_wrapper/main_stereo_vo.cpp ${SRC_STEREO})
target_link_libraries (stereo_vo_node ${catkin_LIBRARIES} ${Eigen_LIBRARIES})



# Additional modules
add_executable(image_cropper 
  src/image_cropper/main_cropper.cpp
  src/image_cropper/image_cropper.cpp)
target_link_libraries (image_cropper ${catkin_LIBRARIES} ${Eigen_LIBRARIES})
