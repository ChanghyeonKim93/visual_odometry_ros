#ifndef _STEREO_VO_ROS1_H_
#define _STEREO_VO_ROS1_H_

#include <iostream>
#include <algorithm>
#include <fstream>

#include <Eigen/Dense>

#include <ros/ros.h>
#include <std_msgs/Float32.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/PointCloud2.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>

#include <image_transport/image_transport.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>

#include <message_filters/sync_policies/approximate_time.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>

#include "visual_odometry_ros/statisticsStamped.h"

// My custom code
#include "core/visual_odometry/stereo_vo/stereo_vo.h"

#include "core/util/timer.h"
#include "core/util/geometry_library.h"

#include "wrapper/ros1/util/ros_print_in_color.h"

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> MySyncPolicy;

class StereoVONode{
public:
    StereoVONode(ros::NodeHandle& nh);
    ~StereoVONode();

private:
    void getParameters();

private:
    void imageStereoCallback(const sensor_msgs::ImageConstPtr &msg_left, const sensor_msgs::ImageConstPtr &msg_right);
    void groundtruthCallback(const geometry_msgs::PoseStampedConstPtr& msg);
    void run();

private:
    void convertPointVecToPointCloud2(const PointVec& X, sensor_msgs::PointCloud2& dst, std::string frame_id);

private:
    ros::NodeHandle nh_;

// Subscribes
    message_filters::Subscriber<sensor_msgs::Image> *left_img_sub_;
    message_filters::Subscriber<sensor_msgs::Image> *right_img_sub_;
    message_filters::Synchronizer<MySyncPolicy> *sync_stereo_;
    
    std::string topicname_image_left_;
    std::string topicname_image_right_;
    
    ros::Subscriber gt_sub_;
    std::string topicname_gt_;


// Publishes
    ros::Publisher pub_pose_;
    std::string topicname_pose_;

    ros::Publisher pub_trajectory_;
    nav_msgs::Path msg_trajectory_;
    std::string topicname_trajectory_;

    ros::Publisher pub_map_points_;
    std::string topicname_map_points_;
    PointVec mappoints_;    

    ros::Publisher pub_statistics_;
    std::string topicname_statistics_;

    ros::Publisher pub_trajectory_gt_;
    nav_msgs::Path msg_trajectory_gt_;
    std::string topicname_trajectory_gt_;

    ros::Publisher pub_debug_image_;
    sensor_msgs::Image msg_debug_image_;

private:
    std::string directory_intrinsic_;

// stereo VO algorithm
private:
    std::unique_ptr<StereoVO> stereo_vo_;

};

#endif