#include <ros/ros.h>
#include <iostream>
#include <time.h>
#include <string>
#include <sstream>
#include <exception>

#include "util/signal_handler_linux.h"

#include "ros_wrapper/stereonode.h"

int main(int argc, char **argv) {
    ros::init(argc, argv, "stereo_vo_node");
    // ros::init(argc, argv, "vo_node", ros::init_options::NoSigintHandler);
    // SignalHandle::initSignalHandler();

    ros::NodeHandle nh("~");
    ROS_INFO_STREAM("stereo_vo_node - starts.");
   
    try {  
        StereoNode node(nh);
    }
    catch (std::exception& e) {
        ROS_ERROR(e.what());
    }
   
    ROS_INFO_STREAM("stereo_vo_node - TERMINATED.");
    return -1;
}