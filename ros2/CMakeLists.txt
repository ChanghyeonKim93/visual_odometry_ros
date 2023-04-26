cmake_minimum_required(VERSION 3.9)
project(visual_odometry_ros2)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)

# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

add_executable(stereo_vo_node visual_odometry/node_stereo_vo.cpp)
ament_target_dependencies(stereo_vo_node rclcpp)
install(TARGETS stereo_vo_node
  DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()
# project(${CMAKE_PROJECT_NAME}_ros2)
# message("   -- [PROJECT]: " ${CMAKE_PROJECT_NAME}_ros2)

# # Default to C++14
# if(NOT CMAKE_CXX_STANDARD)
#   set(CMAKE_CXX_STANDARD 17)
# endif()

# set(CMAKE_BUILD_TYPE Release) # Release, RelWithDebInfo
# # set(CMAKE_CXX_FLAGS "-std=c++17 -O2 -Wall -g -pthread")
# set(CMAKE_CXX_FLAGS "-std=c++17 -O2 -g -pthread")
# set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}) 

# find_package(ament_cmake REQUIRED)
# find_package(rclcpp REQUIRED)
# find_package(std_msgs REQUIRED)

# add_executable(stereo_vo_node visual_odometry/node_stereo_vo.cpp)
# ament_target_dependencies(stereo_vo_node rclcpp std_msgs)
# install(TARGETS stereo_vo_node
#   DESTINATION lib/${CMAKE_PROJECT_NAME})
  
# if(BUILD_TESTING)
#   find_package(ament_lint_auto REQUIRED)
#   # the following line skips the linter which checks for copyrights
#   # uncomment the line when a copyright and license is not present in all source files
#   #set(ament_cmake_copyright_FOUND TRUE)
#   # the following line skips cpplint (only works in a git repo)
#   # uncomment the line when this package is not in a git repo
#   #set(ament_cmake_cpplint_FOUND TRUE)
#   ament_lint_auto_find_test_dependencies()
# endif()
ament_package()