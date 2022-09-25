#include <ros/ros.h>
#include <iostream>
#include <time.h>
#include <string>
#include <sstream>
#include <exception>

#include "image_cropper/image_cropper.h"

int main(int argc, char **argv) {
    ros::init(argc, argv, "image_cropper");

    ros::NodeHandle nh("~");
    ROS_INFO_STREAM("image_cropper - starts.");
    
    int n_cams = 3;
    try 
    {  
        std::shared_ptr<ImageCropper> img_cropper;
        img_cropper = std::make_shared<ImageCropper>(nh);
    }
    catch (std::exception& e) {
        ROS_ERROR(e.what());
    }
   
    ROS_INFO_STREAM("image_cropper - TERMINATED.");
    return -1;
}