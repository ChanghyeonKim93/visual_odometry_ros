#include <ros/ros.h>
#include <iostream>
#include <time.h>
#include <string>
#include <sstream>
#include <exception>

#include "core/util/signal_handler_linux.h"

#include "ros1/visual_odometry/stereo_vo_ros1.h"

int main(int argc, char **argv) {
    ros::init(argc, argv, "stereo_vo_node");
    // ros::init(argc, argv, "vo_node", ros::init_options::NoSigintHandler);
    // SignalHandle::initSignalHandler();

    ros::NodeHandle nh("~");
    ROS_INFO_STREAM("stereo_vo_node - starts.");
   
    try {  
        StereoVONode node(nh);
    }
    catch (std::exception& e) {
        ROS_ERROR(e.what());
    }
   
    ROS_INFO_STREAM("stereo_vo_node - TERMINATED.");
    return -1;
}