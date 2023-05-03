message("   -- [PROJECT]: " ${CMAKE_PROJECT_NAME}_ros2)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_BUILD_TYPE Release) # Release, RelWithDebInfo
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O2 -pthread -pg")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}) 

include_directories(${CMAKE_SOURCE_DIR})
include_directories ("/usr/include/opencv4/")
include_directories(${Eigen3_INCLUDE_DIRS})

set(SRC ${SRC} 
  visual_odometry/stereo_vo_ros2.cpp)
  
add_executable(stereo_vo_node visual_odometry/node_stereo_vo.cpp ${SRC})
target_link_libraries(stereo_vo_node visual_odometry_core ${Eigen3_LIBRARIES} ${OpenCV_LIBS})
ament_target_dependencies(stereo_vo_node rclcpp sensor_msgs geometry_msgs nav_msgs message_filters )
install(TARGETS stereo_vo_node
  DESTINATION lib/${PROJECT_NAME}) # install executable
install(DIRECTORY 
  launch
  DESTINATION share/${PROJECT_NAME}) # install launch files
