project(${CMAKE_PROJECT_NAME}_core)
message("   -- [PROJECT]: " ${CMAKE_PROJECT_NAME}_core)

# Reference:
# https://gist.github.com/luncliff/6e2d4eb7ca29a0afd5b592f72b80cb5c#%EC%86%8C%EC%8A%A4%EC%BD%94%EB%93%9C-%EC%A1%B0%EC%A7%81%ED%99%94

set(CMAKE_BUILD_TYPE Release) # Release, RelWithDebInfo
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O2 -pthread -pg")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}) 

find_package(OpenCV 4 REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories ("/usr/include/opencv4/")
include_directories( ${catkin_INCLUDE_DIRS} )
include_directories( ${Eigen3_INCLUDE_DIRS} )

set(SRC ${SRC}
  util/image_processing.cpp
  util/triangulate_3d.cpp

  util/timer.cpp
  util/signal_handler_linux.cpp
  util/geometry_library.cpp
  util/histogram.cpp

  visual_odometry/feature_extractor.cpp
  visual_odometry/feature_tracker.cpp
  visual_odometry/motion_estimator.cpp

  visual_odometry/ba_solver/sparse_bundle_adjustment.cpp

  visual_odometry/camera.cpp
  visual_odometry/frame.cpp
  visual_odometry/landmark.cpp
  visual_odometry/keyframes.cpp

  visual_odometry/mono_vo/mono_vo.cpp
  visual_odometry/stereo_vo/stereo_vo.cpp
)

# set(CMAKE_INSTALL_PREFIX )
add_library(${CMAKE_PROJECT_NAME}_core STATIC ${SRC})
target_link_libraries(${CMAKE_PROJECT_NAME}_core ${catkin_LIBRARIES} ${Eigen3_LIBRARIES} ${OpenCV_LIBS})
install(TARGETS ${CMAKE_PROJECT_NAME}_core CONFIGURATIONS Release)
  
message("   -- [INFO   ]: DONE")
