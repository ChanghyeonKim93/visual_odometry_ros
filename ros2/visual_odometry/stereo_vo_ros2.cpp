#include "ros2/visual_odometry/stereo_vo_ros2.h"

StereoVONode::StereoVONode(const std::string &node_name)
    : Node(node_name)
{
  std::cerr << "Stereo VO node starts.\n";

  // Get user pamareters
  this->getParameters();

  // Make stereo vo object
  std::string mode = "rosbag";
  stereo_vo_ = std::make_unique<StereoVO>(mode, directory_intrinsic_);

  // Start ROS2 node
  timer_ = this->create_wall_timer(
      10ms, std::bind(&StereoVONode::callbackTimer, this));
  std::cerr << "Stereo VO node wall timer runs.\n";

  rclcpp::QoS image_qos(10);
  image_qos.keep_last(10);
  image_qos.best_effort();
  image_qos.durability_volatile();

  // Subscribers
  sub_img_left_.subscribe(this, topicname_image_left_);
  sub_img_right_.subscribe(this, topicname_image_right_);
  stereo_sync_ = std::make_shared<message_filters::TimeSynchronizer<sensor_msgs::msg::Image, sensor_msgs::msg::Image>>(sub_img_left_, sub_img_right_, 10);
  stereo_sync_->registerCallback(std::bind(&StereoVONode::callbackStereoImages, this, std::placeholders::_1, std::placeholders::_2));
  std::cerr << "Stereo VO node generates subscribers.\n";

  // Publishers
  pub_pose_ = this->create_publisher<nav_msgs::msg::Odometry>(topicname_pose_, 10);
  pub_trajectory_ = this->create_publisher<nav_msgs::msg::Path>(topicname_trajectory_, 10);
  pub_map_points_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(topicname_map_points_, 10);
  pub_debug_image_ = this->create_publisher<sensor_msgs::msg::Image>(topicname_debug_image_,10);
  // pub_debug_image_;
  std::cerr << "Stereo VO node generates publishers.\n";

  std::cerr << "Stereo VO node runs.\n";
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
  std::cerr << "Stereo images are got!\n";
}

void StereoVONode::getParameters()
{
  topicname_image_left_ = "/left/image_raw";
  topicname_image_right_ = "/right/image_raw";
  topicname_pose_ = "/stereo_vo/pose";
  topicname_map_points_ = "/stereo_vo/mappoints";
  topicname_trajectory_ = "/stereo_vo/trajectory";
  topicname_debug_image_ = "/stereo_vo/debug/image";

  directory_intrinsic_ = "/home/kch/ros2_ws/src/visual_odometry_ros/config/stereo/exp_stereo2.yaml";

  if (!get_parameter_or<std::string>("/topicname_image_left", topicname_image_left_, topicname_image_left_))
    std::cerr << "'/topicname_image_left' is not set. Default is " << topicname_image_left_ << "\n";
  if (!get_parameter_or<std::string>("/topicname_image_right", topicname_image_right_, topicname_image_right_))
    std::cerr << "'/topicname_image_right' is not set. Default is " << topicname_image_right_ << "\n";
  if (!get_parameter_or<std::string>("/topicname_pose", topicname_pose_, topicname_pose_))
    std::cerr << "'/topicname_pose' is not set. Default is " << topicname_pose_ << "\n";
  if (!get_parameter_or<std::string>("/topicname_map_points", topicname_map_points_, topicname_map_points_))
    std::cerr << "'/topicname_map_points' is not set. Default is " << topicname_map_points_ << "\n";
  if (!get_parameter_or<std::string>("/topicname_trajectory", topicname_trajectory_, topicname_trajectory_))
    std::cerr << "'/topicname_trajectory' is not set. Default is " << topicname_trajectory_ << "\n";
  if (!get_parameter_or<std::string>("/topicname_debug_image", topicname_debug_image_, "/stereo_vo/debug/image"))
    std::cerr << "'/topicname_debug_image' is not set. Default is " << topicname_debug_image_ << "\n";

  if (!get_parameter_or<std::string>("/directory_intrinsic", directory_intrinsic_, directory_intrinsic_))
    std::cerr << "'/directory_intrinsic' is not set. Default is "<< directory_intrinsic_ << "\n";

  std::cerr << "Stereo VO node gets parameters successfully!\n";
}