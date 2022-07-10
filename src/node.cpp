#include "node.h"

MonoNode::MonoNode(ros::NodeHandle& nh)
: nh_(nh) 
{
    // Get user pamareters
    this->getParameters();

    // Make scale mono vo object
    scale_mono_vo_ = std::make_unique<ScaleMonoVO>("rosbag");

    ROS_INFO_STREAM("MonoNode - generate Scale Mono VO object. Starts.");

    this->run();
};

MonoNode::~MonoNode() { };

void MonoNode::getParameters(){

};

void MonoNode::imageCallback(const sensor_msgs::ImageConstPtr& msg) {
    cv_bridge::CvImageConstPtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvShare(msg);
    } 
    catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    // update camera pose.
    timer::tic();
    // current_frame_time_ = msg->header.stamp;
    // scale_mono_vo_->TrackMonocular(cv_ptr->image, cv_ptr->header.stamp.toSec());
    // Update();
    ROS_INFO_STREAM( "execution time: " << timer::toc(0) << " sec.");
};

void MonoNode::run(){
    ros::Rate rate(200);
    while(nh_.ok()){
        ros::spinOnce();
        rate.sleep();
    }
};
