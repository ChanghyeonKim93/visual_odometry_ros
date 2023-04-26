cmake_minimum_required(VERSION 3.9)
project(${CMAKE_PROJECT_NAME}_ros1)
message("   -- [PROJECT]: " ${CMAKE_PROJECT_NAME}_ros1)

set(CMAKE_BUILD_TYPE Release) # Release, RelWithDebInfo
# set(CMAKE_CXX_FLAGS "-std=c++17 -O2 -Wall -g -pthread")
set(CMAKE_CXX_FLAGS "-std=c++17 -O2 -g -pthread")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}) 

find_package(OpenCV 4 REQUIRED)
MESSAGE("OPENCV VERSION: " ${OpenCV_VERSION}) 

find_package(Eigen3 REQUIRED)
MESSAGE("Eigen3 VERSION: " ${Eigen3_VERSION})

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

catkin_package()

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

# include_directories ("/usr/include/opencv4/")
include_directories(${catkin_INCLUDE_DIRS} )
include_directories(${Eigen3_INCLUDE_DIRS} )

set(LIBS
   ${EIGEN_LIBRARIES}
   ${catkin_LIBRARIES}
   ${OpenCV_LIBS}
)

# executable for 'mono_vo' node
add_executable(mono_vo_node visual_odometry/node_mono_vo.cpp ${SRC_MONO})
target_link_libraries(mono_vo_node visual_odometry_core ${LIBS})

# executable for 'stereo_vo' node
add_executable(stereo_vo_node visual_odometry/node_stereo_vo.cpp ${SRC_STEREO})
target_link_libraries(stereo_vo_node visual_odometry_core ${LIBS})

install(TARGETS
  mono_vo_node
  DESTINATION lib/${PROJECT_NAME})
install(TARGETS
  stereo_vo_node
  DESTINATION lib/${PROJECT_NAME})

message("DEST": ${CATKIN_PACKAGE_SHARE_DESTINATION})
install(DIRECTORY
  launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
 )