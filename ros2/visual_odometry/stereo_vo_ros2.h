#ifndef _STEREO_VO_ROS2_H_
#define _STEREO_VO_ROS2_H_

#include <iostream>
#include <memory>
#include <algorithm>
#include <fstream>
#include <string>
#include <cstring>

#include <eigen3/Eigen/Dense>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
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
  StereoVONode();
  ~StereoVONode();

private:
  void callbackStereoImages(
    const sensor_msgs::msg::Image::ConstSharedPtr &msg_left, 
    const sensor_msgs::msg::Image::ConstSharedPtr &msg_right) const;
  
  void callbackTimer();
    
private:
  rclcpp::TimerBase::SharedPtr timer_;

  message_filters::Subscriber<sensor_msgs::msg::Image> sub_img_left_;
  message_filters::Subscriber<sensor_msgs::msg::Image> sub_img_right_;
  std::shared_ptr<message_filters::TimeSynchronizer<sensor_msgs::msg::Image, sensor_msgs::msg::Image>> stereo_sync_;
};

#endif
