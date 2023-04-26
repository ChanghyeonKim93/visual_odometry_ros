#include <ros/ros.h>
#include <iostream>
#include <time.h>
#include <string>
#include <sstream>
#include <exception>
#include <memory>

#include "wrapper/ros1/stereo_rectifier/stereo_rectifier.h"

int main(int argc, char **argv)
{
    ros::init(argc, argv, "stereo_rectifier");

    ros::NodeHandle nh("~");
    ROS_INFO_STREAM("stereo_rectifier - starts.");

    try
    {
        std::unique_ptr<StereoRectifier> stereo_rectifier =
            std::make_unique<StereoRectifier>(nh);
    }
    catch (std::exception &e)
    {
        ROS_ERROR(e.what());
    }

    ROS_INFO_STREAM("stereo_rectifier - TERMINATED.");
    return -1;
}