#ifndef _NODE_H_
#define _NODE_H_

#include <iostream>
#include <algorithm>
#include <fstream>

#include <Eigen/Dense>

#include <ros/ros.h>
#include <sensor_msgs/Image.h>

#include <nav_msgs/Odometry.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>

#include "core/scale_mono_vo.h"

#include "util/timer.h"

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
    
    ros::Subscriber img_sub_;
    std::string topicname_image_;

    ros::Publisher pub_pose_estimation_;
    std::string topicname_pose_estimation_;

    std::string directory_intrinsic_;

private:
    std::unique_ptr<ScaleMonoVO> scale_mono_vo_;
};

#endif