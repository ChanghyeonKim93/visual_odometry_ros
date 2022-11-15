#ifndef _STEREO_RECTIFIER_H_
#define _STEREO_RECTIFIER_H_

#include <iostream>
#include <algorithm>
#include <fstream>

#include <Eigen/Dense>

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>

#include <image_transport/image_transport.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>

#include <message_filters/sync_policies/approximate_time.h>


#include <cv_bridge/cv_bridge.h>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// Defines 
#include "core/defines.h"
#include "core/type_defines.h"

// custom
#include "core/camera.h"
#include "core/image_processing.h"

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> MySyncPolicy;

class StereoRectifier
{
private:
    StereoCameraPtr stereo_cam_;
    ros::NodeHandle nh_;


    std::string directory_intrinsic_;

private:
    // Subscribes
    message_filters::Subscriber<sensor_msgs::Image> *left_img_sub_;
    message_filters::Subscriber<sensor_msgs::Image> *right_img_sub_;
    message_filters::Synchronizer<MySyncPolicy> *sync_stereo_;
    
    std::string topicname_image_left_;
    std::string topicname_image_right_;



    ros::Publisher pub_left_rect_;
    ros::Publisher pub_right_rect_;
    std::string topicname_image_left_rect_;
    std::string topicname_image_right_rect_;

private:
    void imageStereoCallback(
        const sensor_msgs::ImageConstPtr &msg_left, const sensor_msgs::ImageConstPtr &msg_right);

    void run();

public:
    StereoRectifier(ros::NodeHandle& nh);
    void loadStereoCameraIntrinsics(const std::string& dir);

};
#endif