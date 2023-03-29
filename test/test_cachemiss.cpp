#include <iostream>
#include <vector>

#include <string>
#include <sstream>
#include <exception>

#include <Eigen/Dense>

#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>

int main(int argc, char **argv) {
    ros::init(argc, argv, "test_cachemiss");
    ros::NodeHandle nh("~");
    
    ROS_INFO_STREAM("test_cachemiss - starts.");
   
    try {  
        std::cout << "RUN!\n";

        cv::Mat img;
    }
    catch (std::exception& e) {
        ROS_ERROR(e.what());
    }
   
    ROS_INFO_STREAM("test_cachemiss - TERMINATED.");
    return -1;
}