#ifndef _NODE_H_
#define _NODE_H_

#include <iostream>
#include <algorithm>
#include <fstream>

#include <Eigen/Dense>

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/PointCloud2.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>

#include "core/scale_mono_vo.h"

#include "util/timer.h"

#include "scale_mono_vo_ros/statisticsStamped.h"

class MonoNode{
public:
    MonoNode(ros::NodeHandle& nh);
    ~MonoNode();

private:
    void getParameters();

private:
    void imageCallback(const sensor_msgs::ImageConstPtr& msg);
    void run();

private:
    ros::NodeHandle nh_;
    
    // subscriber
    ros::Subscriber img_sub_;
    std::string topicname_image_;

    // publishers
    ros::Publisher pub_pose_;
    std::string topicname_pose_;

    ros::Publisher pub_trajectory_;
    std::string topicname_trajectory_;

    ros::Publisher pub_map_points_;
    std::string topicname_map_points_;

    ros::Publisher pub_statistics_;
    std::string topicname_statistics_;

private:
    std::string directory_intrinsic_;

private:
    std::unique_ptr<ScaleMonoVO> scale_mono_vo_;
};

#endif