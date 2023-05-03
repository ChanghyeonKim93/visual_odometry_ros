#include <iostream>
#include <string>
#include <exception>

#include "ros2/visual_odometry/stereo_vo_ros2.h"

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  try{
    std::string node_name = "stereo_vo_node";
    rclcpp::spin(std::make_shared<StereoVONode>(node_name));
    rclcpp::shutdown();
  }
  catch (std::exception& e)
  {
    std::cerr << e.what() << std::endl;
  }

  return 0;
}
