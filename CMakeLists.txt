cmake_minimum_required(VERSION 3.8)
project(visual_odometry_ros)
set(CMAKE_BUILD_TYPE Release) # Release, RelWithDebInfo
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

# Source files for the mono VO
set(SRC_MONO ${SRC_MONO} 
  src/ros_wrapper/mono_vo_node.cpp
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
  
  src/core/mono_vo/mono_vo.cpp
  src/core/mono_vo/mono_vo_trackimage.cpp
)

# Source files for the stereo VO
set(SRC_STEREO ${SRC_STEREO} 
  src/ros_wrapper/stereo_vo_node.cpp
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

# executable for 'mono_vo' node
add_executable(mono_vo_node src/ros_wrapper/main_mono_vo.cpp ${SRC_MONO})
target_link_libraries (mono_vo_node ${catkin_LIBRARIES} ${Eigen_LIBRARIES})

# executable for 'stereo_vo' node
add_executable(stereo_vo_node src/ros_wrapper/main_stereo_vo.cpp ${SRC_STEREO})
target_link_libraries (stereo_vo_node ${catkin_LIBRARIES} ${Eigen_LIBRARIES})

# # executable for 'test orb matching' node
# add_executable(test_orb_matching_node src/ros_wrapper/main_orbmatching_test.cpp ${SRC_MONO})
# target_link_libraries (test_orb_matching_node ${catkin_LIBRARIES} ${Eigen_LIBRARIES})


# executable for the additional modules
add_executable(image_cropper 
  src/image_cropper/main_cropper.cpp
  src/image_cropper/image_cropper.cpp)
target_link_libraries (image_cropper ${catkin_LIBRARIES} ${Eigen_LIBRARIES})

add_executable(stereo_rectifier 
  src/stereo_rectifier/main_stereo_rectifier.cpp
  src/core/camera.cpp
  src/core/image_processing.cpp
  src/stereo_rectifier/stereo_rectifier.cpp)
target_link_libraries (stereo_rectifier ${catkin_LIBRARIES} ${Eigen_LIBRARIES})

add_executable(test_pose3d 
  src/test/main_test_pose3d.cpp
  src/util/pose3d.cpp
)
target_link_libraries (test_pose3d ${catkin_LIBRARIES} ${Eigen_LIBRARIES})

