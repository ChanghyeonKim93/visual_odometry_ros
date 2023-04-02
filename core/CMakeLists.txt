project(${CMAKE_PROJECT_NAME}_core)
message("   -- [PROJECT]: " ${CMAKE_PROJECT_NAME}_core)

# Reference:
# https://gist.github.com/luncliff/6e2d4eb7ca29a0afd5b592f72b80cb5c#%EC%86%8C%EC%8A%A4%EC%BD%94%EB%93%9C-%EC%A1%B0%EC%A7%81%ED%99%94

set(CMAKE_BUILD_TYPE Release) # Release, RelWithDebInfo
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O2 -pthread -pg")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}) 

find_package (OpenCV 4.0.0 REQUIRED)
include_directories ("/usr/include/opencv4/")

find_package(Eigen3 REQUIRED)

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

# ADD_LIBRARY ( <라이브러리_이름> [STATIC|SHARED|MODULE] <소스_파일> <소스_파일> ... )
add_library(${CMAKE_PROJECT_NAME}_core SHARED ${SRC})
target_link_libraries(${CMAKE_PROJECT_NAME}_core ${catkin_LIBRARIES} ${Eigen3_LIBRARIES})

# <라이브러리_이름> : 생성할 라이브러리 이름 (lib~.a / lib~.so 에서 ~에 들어갈 값)
# [STATIC|SHARED|MODULE] : 라이브러리 종류 (생략시 STATIC)
# <소스_파일> : 라이브러리를 생성하는 데 필요한 소스 파일

# INSTALL ( TARGETS <Target_목록>
#         RUNTIME DESTINATION <바이너리_설치_경로>
#         LIBRARY DESTINATION <라이브러리_설치_경로>
#         ARCHIVE DESTINATION <아카이브_설치_경로>
#         )
# SET (CMAKE_INSTALL_PREFIX ~)
install(TARGETS ${CMAKE_PROJECT_NAME}_core
  DESTINATION lib)

message("   -- [INFO   ]: DONE")
