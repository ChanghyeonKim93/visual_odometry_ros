project(${CMAKE_PROJECT_NAME}_ros1)
message("   -- [PROJECT]: " ${CMAKE_PROJECT_NAME}_ros1)
# message(${CMAKE_CURRENT_SOURCE_DIR}/../core/)
# message(${CMAKE_SOURCE_DIR} )

set(CMAKE_BUILD_TYPE Release) # Release, RelWithDebInfo
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O2 -pthread -pg")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}) 

find_package(catkin REQUIRED cmake_modules)
find_package(OpenCV 4 REQUIRED)
find_package(Eigen3 REQUIRED)

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
  DEPENDS Eigen3
  CATKIN_DEPENDS roscpp message_runtime
)

set(INCLUDE_DIRS ${INCLUDE_DIRS} ${PROJECT_SOURCE_DIR})

# Source files for the mono VO
set(SRC_MONO ${SRC_MONO} 
  visual_odometry/mono_vo_ros1.cpp
  util/ros_print_in_color.cpp
)

# Source files for the stereo VO
set(SRC_STEREO ${SRC_STEREO} 
  visual_odometry/stereo_vo_ros1.cpp
  util/ros_print_in_color.cpp
)

set(INCLUDE_DIRS ${INCLUDE_DIRS} ${CMAKE_CURRENT_BINARY_DIR} )

# include_directories( ${CMAKE_SOURCE_DIR})
# include_directories( ${CMAKE_SOURCE_DIR}/core)
# include_directories( ${CMAKE_SOURCE_DIR}/test)
# include_directories( ${CMAKE_SOURCE_DIR}/ros1)

include_directories ("/usr/include/opencv4/")
include_directories( ${catkin_INCLUDE_DIRS} )
include_directories( ${Eigen3_INCLUDE_DIRS} )

# Print some message showing some of them

link_directories(/usr/local/lib)

# executable for 'mono_vo' node
add_executable(mono_vo_node visual_odometry/node_mono_vo.cpp ${SRC_MONO})
target_link_libraries(mono_vo_node ~/ros2_ws/build/visual_odometry/core/libvisual_odometry_core.a)
# target_include_directories(mono_vo_node)
# target_link_libraries(mono_vo_node ${catkin_LIBRARIES} ${Eigen3_LIBRARIES})
# target_link_libraries(mono_vo_node visual_odometry_core)

# executable for 'stereo_vo' node
add_executable(stereo_vo_node visual_odometry/node_stereo_vo.cpp ${SRC_STEREO})
target_link_libraries(stereo_vo_node ~/ros2_ws/build/visual_odometry/core/libvisual_odometry_core.a)
# target_link_libraries(stereo_vo_node visual_odometry_core)


# # executable for 'test programs'
# add_executable(test_alignedmemory test/test_alignedmemory.cpp test/aligned_memory.h)

# add_executable(test_cachemiss test/test_cachemiss.cpp core/util/timer.cpp)
# target_link_libraries(test_cachemiss ${catkin_LIBRARIES} ${Eigen_LIBRARIES})

# executable for the additional modules
# add_executable(image_cropper 
#   src/image_cropper/main_cropper.cpp
#   src/image_cropper/image_cropper.cpp)
# target_link_libraries (image_cropper ${catkin_LIBRARIES} ${Eigen_LIBRARIES})

# add_executable(stereo_rectifier 
#   src/stereo_rectifier/main_stereo_rectifier.cpp
#   src/core/camera.cpp
#   src/core/image_processing.cpp
#   src/stereo_rectifier/stereo_rectifier.cpp)
# target_link_libraries (stereo_rectifier ${catkin_LIBRARIES} ${Eigen_LIBRARIES})

# add_executable(test_pose3d 
#   src/test/main_test_pose3d.cpp
#   src/util/pose3d.cpp
# )
# target_link_libraries (test_pose3d ${catkin_LIBRARIES} ${Eigen_LIBRARIES})

