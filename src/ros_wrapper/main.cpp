#include <ros/ros.h>
#include <iostream>
#include <time.h>
#include <string>
#include <sstream>
#include <exception>

#include "util/signal_handler_linux.h"

#include "mononode.h"

int main(int argc, char **argv) {
    ros::init(argc, argv, "vo_node");
    // ros::init(argc, argv, "vo_node", ros::init_options::NoSigintHandler);
    // SignalHandle::initSignalHandler();

    ros::NodeHandle nh("~");
    ROS_INFO_STREAM("vo_node - starts.");
   
    try {  
        MonoNode node(nh);
    }
    catch (std::exception& e) {
        ROS_ERROR(e.what());
    }
   
    ROS_INFO_STREAM("vo_node - TERMINATED.");
    return -1;
}