#include "ros2/visual_odometry/stereo_vo_ros2.h"

int encoding2mat_type(const std::string &encoding)
{
  if (encoding == "mono8")
  {
    return CV_8UC1;
  }
  else if (encoding == "bgr8")
  {
    return CV_8UC3;
  }
  else if (encoding == "mono16")
  {
    return CV_16SC1;
  }
  else if (encoding == "rgba8")
  {
    return CV_8UC4;
  }
  else if (encoding == "bgra8")
  {
    return CV_8UC4;
  }
  else if (encoding == "32FC1")
  {
    return CV_32FC1;
  }
  else if (encoding == "rgb8")
  {
    return CV_8UC3;
  }
  else
  {
    throw std::runtime_error("Unsupported encoding type");
  }
}

std::string mat_type2encoding(int mat_type)
{
  switch (mat_type)
  {
  case CV_8UC1:
    return "mono8";
  case CV_8UC3:
    return "bgr8";
  case CV_16SC1:
    return "mono16";
  case CV_8UC4:
    return "rgba8";
  default:
    throw std::runtime_error("Unsupported encoding type");
  }
}

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
  pub_debug_image_ = this->create_publisher<sensor_msgs::msg::Image>(topicname_debug_image_, 10);
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
  rclcpp::Time t_callback_start = this->get_clock()->now();

  cv::Mat image_left(msg_left->height, msg_left->width, encoding2mat_type(msg_left->encoding),
                     const_cast<unsigned char *>(msg_left->data.data()), msg_left->step);
  cv::Mat image_right(msg_right->height, msg_right->width, encoding2mat_type(msg_right->encoding),
                      const_cast<unsigned char *>(msg_right->data.data()), msg_right->step);

  // update camera pose.
  rclcpp::Time t_track_start = this->get_clock()->now();
  double time_now = static_cast<double>(msg_left->header.stamp.sec) + static_cast<double>(msg_left->header.stamp.nanosec) * 1e-9;
  stereo_vo_->trackStereoImages(image_left, image_right, time_now);
  rclcpp::Time t_track_end = this->get_clock()->now();

  double time_track_start = static_cast<double>(t_track_start.seconds()) + static_cast<double>(t_track_start.nanoseconds()) * 1e-9;
  double time_track_end = static_cast<double>(t_track_end.seconds()) + static_cast<double>(t_track_end.nanoseconds()) * 1e-9;
  std::cerr << "Time for track: " << (time_track_end - time_track_start) * 1000.0 << " [ms]\n";

  // Show statistics & get odometry results
  const StereoVO::AlgorithmStatistics &stat = stereo_vo_->getStatistics();

  // Pose publish
  nav_msgs::msg::Odometry msg_pose;
  msg_pose.header.stamp = this->get_clock()->now();
  msg_pose.header.frame_id = "map";

  const PoseSE3 &Twc = stat.stats_frame.back().Twc;
  Eigen::Vector4f q = geometry::r2q_f(Twc.block<3, 3>(0, 0));
  msg_pose.pose.pose.position.x = Twc(0, 3);
  msg_pose.pose.pose.position.y = Twc(1, 3);
  msg_pose.pose.pose.position.z = Twc(2, 3);
  msg_pose.pose.pose.orientation.w = q(0);
  msg_pose.pose.pose.orientation.x = q(1);
  msg_pose.pose.pose.orientation.y = q(2);
  msg_pose.pose.pose.orientation.z = q(3);

  pub_pose_->publish(msg_pose);

  // Publish path
  msg_trajectory_.header.frame_id = "map";
  msg_trajectory_.header.stamp = this->get_clock()->now();

  geometry_msgs::msg::PoseStamped p;
  p.header.frame_id = "map";
  p.header.stamp = this->get_clock()->now();
  p.pose = msg_pose.pose.pose;
  msg_trajectory_.poses.push_back(p);

  msg_trajectory_.poses.resize(stat.stats_keyframe.size());
  for (int j = 0; j < msg_trajectory_.poses.size(); ++j)
  {
    PoseSE3 Twc = stat.stats_keyframe[j].Twc;
    msg_pose.pose.pose.position.x = Twc(0, 3);
    msg_pose.pose.pose.position.y = Twc(1, 3);
    msg_pose.pose.pose.position.z = Twc(2, 3);
    msg_trajectory_.poses[j].pose = msg_pose.pose.pose;
  }

  pub_trajectory_->publish(msg_trajectory_);

  // Publish mappoints
  sensor_msgs::msg::PointCloud2 msg_mappoint;
  size_t cnt_total_pts = 0;
  for (size_t j = 0; j < stat.stats_keyframe.size(); ++j)
  {
    cnt_total_pts += stat.stats_keyframe[j].mappoints.size();
  }
  mappoints_.resize(cnt_total_pts);
  cnt_total_pts = 0;
  for (size_t j = 0; j < stat.stats_keyframe.size(); ++j)
  {
    for (const auto &x : stat.stats_keyframe[j].mappoints)
    {
      mappoints_[cnt_total_pts] = x;
      ++cnt_total_pts;
    }
  }
  convertPointVecToPointCloud2(mappoints_, msg_mappoint, "map");
  pub_map_points_->publish(msg_mappoint);
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

  this->declare_parameter<std::string>("topicname_image_left", "-");
  this->declare_parameter<std::string>("topicname_image_right", "-");
  this->declare_parameter<std::string>("topicname_pose", "-");
  this->declare_parameter<std::string>("topicname_map_points", "-");
  this->declare_parameter<std::string>("topicname_trajectory", "-");
  this->declare_parameter<std::string>("topicname_debug_image", "-");
  this->declare_parameter<std::string>("directory_intrinsic", "-");

  if (!this->get_parameter_or<std::string>("topicname_image_left", topicname_image_left_, topicname_image_left_))
    std::cerr << "'topicname_image_left' is not set. Default is " << topicname_image_left_ << "\n";
  else
    std::cerr << "'topicname_image_left' is " << topicname_image_left_ << "\n";

  if (!this->get_parameter_or<std::string>("topicname_image_right", topicname_image_right_, topicname_image_right_))
    std::cerr << "'topicname_image_right' is not set. Default is " << topicname_image_right_ << "\n";
  else
    std::cerr << "'topicname_image_right' is " << topicname_image_right_ << "\n";

  if (!this->get_parameter_or<std::string>("topicname_pose", topicname_pose_, topicname_pose_))
    std::cerr << "'topicname_pose' is not set. Default is " << topicname_pose_ << "\n";
  else
    std::cerr << "'topicname_pose' is " << topicname_pose_<< "\n";

  if (!this->get_parameter_or<std::string>("topicname_map_points", topicname_map_points_, topicname_map_points_))
    std::cerr << "'topicname_map_points' is not set. Default is " << topicname_map_points_ << "\n";
  else
    std::cerr << "'topicname_map_points' is " << topicname_map_points_<< "\n";

  if (!this->get_parameter_or<std::string>("topicname_trajectory", topicname_trajectory_, topicname_trajectory_))
    std::cerr << "'topicname_trajectory' is not set. Default is " << topicname_trajectory_ << "\n";
  else
    std::cerr << "'topicname_trajectory' is " << topicname_trajectory_<< "\n";

  if (!this->get_parameter_or<std::string>("topicname_debug_image", topicname_debug_image_, topicname_debug_image_))
    std::cerr << "'topicname_debug_image' is not set. Default is " << topicname_debug_image_ << "\n";
  else
    std::cerr << "'topicname_debug_image' is " << topicname_debug_image_<< "\n";

  if (!this->get_parameter_or<std::string>("directory_intrinsic", directory_intrinsic_, directory_intrinsic_))
    std::cerr << "'directory_intrinsic' is not set. Default is " << directory_intrinsic_ << "\n";
  else
    std::cerr << "'directory_intrinsic' is " << directory_intrinsic_<< "\n";

  std::cerr << "Stereo VO node gets parameters successfully!\n";
}

void StereoVONode::convertPointVecToPointCloud2(const PointVec &X, sensor_msgs::msg::PointCloud2 &dst, std::string frame_id)
{
  size_t n_pts = X.size();

  // intensity mapping (-3 m ~ 3 m to 0~255)
  float z_min = -3.0;
  float z_max = 3.0;
  float intensity_min = 30;
  float intensity_max = 255;
  float slope = (intensity_max - intensity_min) / (z_max - z_min);

  dst.header.frame_id = frame_id;
  dst.header.stamp = this->get_clock()->now();
  // ROS_INFO_STREAM(dst.header.stamp << endl);
  dst.width = n_pts;
  dst.height = 1;

  sensor_msgs::msg::PointField f_tmp;
  f_tmp.offset = 0;
  f_tmp.name = "x";
  f_tmp.datatype = sensor_msgs::msg::PointField::FLOAT32;
  dst.fields.push_back(f_tmp);
  f_tmp.offset = 4;
  f_tmp.name = "y";
  f_tmp.datatype = sensor_msgs::msg::PointField::FLOAT32;
  dst.fields.push_back(f_tmp);
  f_tmp.offset = 8;
  f_tmp.name = "z";
  f_tmp.datatype = sensor_msgs::msg::PointField::FLOAT32;
  dst.fields.push_back(f_tmp);
  f_tmp.offset = 12;
  f_tmp.name = "intensity";
  f_tmp.datatype = sensor_msgs::msg::PointField::FLOAT32;
  dst.fields.push_back(f_tmp);
  f_tmp.offset = 16;
  f_tmp.name = "ring";
  f_tmp.datatype = sensor_msgs::msg::PointField::UINT16;
  dst.fields.push_back(f_tmp);
  f_tmp.offset = 18;
  f_tmp.name = "time";
  f_tmp.datatype = sensor_msgs::msg::PointField::FLOAT32;
  dst.fields.push_back(f_tmp);
  dst.point_step = 22; // x 4 + y 4 + z 4 + i 4 + r 2 + t 4

  dst.data.resize(dst.point_step * dst.width);
  for (size_t i = 0; i < dst.width; ++i)
  {
    size_t i_ptstep = i * dst.point_step;
    size_t arrayPosX = i_ptstep + dst.fields[0].offset; // X has an offset of 0
    size_t arrayPosY = i_ptstep + dst.fields[1].offset; // Y has an offset of 4
    size_t arrayPosZ = i_ptstep + dst.fields[2].offset; // Z has an offset of 8

    size_t ind_intensity = i_ptstep + dst.fields[3].offset; // 12
    size_t ind_ring = i_ptstep + dst.fields[4].offset;      // 16
    size_t ind_time = i_ptstep + dst.fields[5].offset;      // 18

    float height_intensity = slope * (X[i](2) - z_min) + intensity_min;
    if (height_intensity >= intensity_max)
      height_intensity = intensity_max;
    if (height_intensity <= intensity_min)
      height_intensity = intensity_min;

    float x = X[i](0);
    float y = X[i](1);
    float z = X[i](2);

    memcpy(&dst.data[arrayPosX], &(x), sizeof(float));
    memcpy(&dst.data[arrayPosY], &(y), sizeof(float));
    memcpy(&dst.data[arrayPosZ], &(z), sizeof(float));
    memcpy(&dst.data[ind_intensity], &(height_intensity), sizeof(float));
    memcpy(&dst.data[ind_ring], &(x), sizeof(unsigned short));
    memcpy(&dst.data[ind_time], &(x), sizeof(float));
  }
}