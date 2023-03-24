#include <iostream>
#include <time.h>
#include <string>
#include <sstream>
#include <exception>

#include <ros/ros.h>

#include "core/util/signal_handler_linux.h"

#include "wrapper/ros1/visual_odometry/mono_vo_node.h"

int main(int argc, char **argv) {
    ros::init(argc, argv, "mono_vo_node");
    // ros::init(argc, argv, "vo_node", ros::init_options::NoSigintHandler);
    // SignalHandle::initSignalHandler();

    ros::NodeHandle nh("~");
    ROS_INFO_STREAM("mono_vo_node - starts.");
   
    try {  
        MonoVONode node(nh);
    }
    catch (std::exception& e) {
        ROS_ERROR(e.what());
    }
   
    ROS_INFO_STREAM("mono_vo_node - TERMINATED.");
    return -1;
}