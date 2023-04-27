cmake_minimum_required(VERSION 3.10)
project(visual_odometry_ros2)
message("   -- [PROJECT]: " visual_odometry_ros2)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)

add_executable(stereo_vo_node visual_odometry/node_stereo_vo.cpp)
ament_target_dependencies(stereo_vo_node rclcpp)
install(TARGETS stereo_vo_node
  DESTINATION lib/${PROJECT_NAME})
ament_package()