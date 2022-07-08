#ifndef _NODE_H_
#define _NODE_H_

#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>

#include <Eigen/Dense>

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>

#include "scale_mono_vo.h"

#include "timer.h"

class MonoNode{

public:
    MonoNode(ros::NodeHandle& nh);
    ~MonoNode();


private:
    void getParameters();
    void imageCallback(const sensor_msgs::ImageConstPtr& msg);

    void run();

private:
    ros::NodeHandle nh_;
    image_transport::Subscriber img_sub_;

private:
    std::unique_ptr<ScaleMonoVO> scale_mono_vo_;
};

#endif