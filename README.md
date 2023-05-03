# visual_odometry_ros
This repository includes 'monocular visual odometry' and 'stereo visual odometry' wrapped by ROS1 and ROS2 (to be updated soon).

*  **stereo_vo_node:** This module is a stereo visual odometry node. 
   - It requires streaming 'time-synchronized' stereo images. 
   - This module yields the metric-scale camera motion estimations in real-time.

*  **mono_vo_node:** This module is a monocular visual odometry node. 
   - It only requires streaming monocular images. 
   - This module yields the up-to-scale camera motion estimations in real-time.

## Installation
### 1. git clone
```
cd ~/{YOUR_ROS_WS}/src
git clone "https://github.com/ChanghyeonKim93/visual_odometry_ros.git"
```
### 2. build the library & install the library files
```
cd ~/{YOUR_ROS_WS}/src/visual_odometry_ros
mkdir build
cd build
cmake .. && make -j8
sudo make install -y
```
### 3. catkin (colcon) build to make ROS1 (or 2) nodes
```
cd ~/{YOUR_ROS_WS}
catkin build visual_odometry_ros (or colcon build --base-path src/visual_odometry_ros)
(for ROS2 only) source install/local_setup.bash && source install/setup.bash
```

## Run
```
cd ~/{YOUR_ROS_WS}
ros2 launch visual_odometry stereo_vo.launch.py
```