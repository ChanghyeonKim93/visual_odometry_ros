cmake_minimum_required(VERSION 3.9)
SET(CMAKE_PROJECT_NAME visual_odometry)
project(${CMAKE_PROJECT_NAME})
message("-- [PROJECT]: " ${CMAKE_PROJECT_NAME})

message("-- [INFO   ]: Processing 'visual_odometry' project...")
message("-- [INFO   ]: Go to ros1/CMakeLists.txt ...")
message("-- [INFO   ]: Processing ros1/CMakeLists.txt ...")

# add_compile_options( -g -Wall )

if(WIN32)
  message("-- [INFO   ]: SYSTEM TYPE: WIN32")
elseif(UNIX)
  message("-- [INFO   ]: SYSTEM TYPE: UNIX")
endif()

include_directories(${CMAKE_SOURCE_DIR})

add_subdirectory(core)
add_subdirectory(ros1)
# add_subdirectory(ros2)
