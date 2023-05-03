#include "ros2/visual_odometry/stereo_vo_ros2.h"

StereoVONode::StereoVONode(const std::string& node_name)
    : Node(node_name)
{
  timer_ = this->create_wall_timer(
      10ms, std::bind(&StereoVONode::callbackTimer, this));
  std::cerr << "stereo node is initialized.\n";

  rclcpp::QoS image_qos(10);
  image_qos.keep_last(10);
  image_qos.best_effort();
  image_qos.durability_volatile();

  sub_img_left_.subscribe(this, "left/image_raw");
  sub_img_right_.subscribe(this, "right/image_raw");

  stereo_sync_ = std::make_shared<message_filters::TimeSynchronizer<sensor_msgs::msg::Image, sensor_msgs::msg::Image>>(sub_img_left_, sub_img_right_, 10);
  stereo_sync_->registerCallback(std::bind(&StereoVONode::callbackStereoImages, this, std::placeholders::_1, std::placeholders::_2));
}

StereoVONode::~StereoVONode()
{
}

void StereoVONode::callbackTimer()
{
  // std::cerr << "Node is run" << std::endl;
}

void StereoVONode::callbackStereoImages(
    const sensor_msgs::msg::Image::ConstSharedPtr &msg_left,
    const sensor_msgs::msg::Image::ConstSharedPtr &msg_right)
{
  msg_left;
  msg_right;

  std::cerr << "Stereo Image is got!\n";
}