#include <iostream>
#include "ros2/visual_odometry/stereo_vo_ros2.h"

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<StereoVONode>());
  rclcpp::shutdown();
  return 0;
}
