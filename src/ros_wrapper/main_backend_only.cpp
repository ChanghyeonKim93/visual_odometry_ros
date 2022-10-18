#include <ros/ros.h>
#include <iostream>
#include <time.h>
#include <string>
#include <sstream>
#include <exception>

#include "util/signal_handler_linux.h"

#include "ros_wrapper/backend_node.h"

int main(int argc, char **argv) {
    ros::init(argc, argv, "backend_node");
    ros::NodeHandle nh("~");
    
    ROS_INFO_STREAM("backend_node - starts.");
   
    try {  
        BackendNode node(nh);
    }
    catch (std::exception& e) {
        ROS_ERROR(e.what());
    }
   
    ROS_INFO_STREAM("backend_node - TERMINATED.");
    return -1;
}