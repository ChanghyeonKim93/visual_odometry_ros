#ifndef _STEREO_VO_ROS2_H_
#define _STEREO_VO_ROS2_H_

#include <iostream>
#include <memory>
#include <algorithm>
#include <fstream>
#include <string>
#include <cstring>
#include <exception>

#include <eigen3/Eigen/Dense>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "nav_msgs/msg/path.hpp"

#include "message_filters/subscriber.h"
// #include <message_filters/synchronizer.h>
#include "message_filters/sync_policies/approximate_time.h"
#include "message_filters/time_synchronizer.h"

#include <opencv2/core/core.hpp>

// My custom code
#include "core/visual_odometry/stereo_vo/stereo_vo.h"

#include "core/util/timer.h"
#include "core/util/geometry_library.h"

using namespace std::chrono_literals;

class StereoVONode : public rclcpp::Node
{
public:
  StereoVONode(const std::string& node_name);
  ~StereoVONode();

private:
  void getParameters();

private:
  void callbackStereoImages(
    const sensor_msgs::msg::Image::ConstSharedPtr &msg_left, 
    const sensor_msgs::msg::Image::ConstSharedPtr &msg_right);
  
  void callbackTimer();

private:
  void convertPointVecToPointCloud2(const PointVec& X, sensor_msgs::msg::PointCloud2& dst, std::string frame_id);
    
private:
  rclcpp::TimerBase::SharedPtr timer_;

  message_filters::Subscriber<sensor_msgs::msg::Image> sub_img_left_;
  message_filters::Subscriber<sensor_msgs::msg::Image> sub_img_right_;
  std::shared_ptr<message_filters::TimeSynchronizer<sensor_msgs::msg::Image, sensor_msgs::msg::Image>> stereo_sync_;

  std::string topicname_image_left_;
  std::string topicname_image_right_;

  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_pose_;
  std::string topicname_pose_;

  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_trajectory_;
  std::string topicname_trajectory_;
  nav_msgs::msg::Path msg_trajectory_;

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_map_points_;
  std::string topicname_map_points_;
  PointVec mappoints_;

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_debug_image_;
  std::string topicname_debug_image_;
  sensor_msgs::msg::Image msg_debug_image_;

private:
  std::string directory_intrinsic_;

// stereo VO algorithm
private:
    std::unique_ptr<StereoVO> stereo_vo_;
};

#endif
