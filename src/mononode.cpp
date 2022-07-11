#include "mononode.h"

/**
 * @brief MonoNode 생성자. ROS wrapper for scale mono vo.
 * @details In this function, ROS parameters are get by 'getParameters()'. 
 *          Then, scale_mono_vo object is constructed, and 'run()' function is called.
 * @param nh ros::Nodehandle. 
 * @return none
 * @author Changhyeon Kim (hyun91015@gmail.com)
 * @date 10-July-2022
 */
MonoNode::MonoNode(ros::NodeHandle& nh) : nh_(nh) 
{
    // Get user pamareters
    this->getParameters();

    // Make scale mono vo object
    std::string mode = "rosbag";
    scale_mono_vo_ = std::make_unique<ScaleMonoVO>(mode, directory_intrinsic_);

    // Subscriber    
    img_sub_ = 
        nh_.subscribe<sensor_msgs::Image>(topicname_image_, 10, &MonoNode::imageCallback, this);

    // Publisher
    pub_pose_estimation_ = 
        nh_.advertise<nav_msgs::Odometry>(topicname_pose_estimation_, 1);

    ROS_INFO_STREAM("MonoNode - generate Scale Mono VO object. Starts.");

    // spin .
    this->run();
};

/**
 * @brief MonoNode 소멸자.
 * @details MonoNode 소멸자. 
 * @return none
 * @author Changhyeon Kim (hyun91015@gmail.com)
 * @date 10-July-2022
 */
MonoNode::~MonoNode(){

};

/**
 * @brief function to get the ROS parameters from the launch file.
 * @details function to get the ROS parameters from the launch file. 만약 파라미터가 세팅되지 않았다면, runtime_error를 throw 함.
 * @return none
 * @author Changhyeon Kim (hyun91015@gmail.com)
 * @date 10-July-2022
 */
void MonoNode::getParameters(){
    if(!ros::param::has("~topicname_image"))
        throw std::runtime_error("'topicname_image' is not set.");
    if(!ros::param::has("~topicname_pose_estimation"))
        throw std::runtime_error("'topicname_pose_estimation' is not set.");
    if(!ros::param::has("~directory_intrinsic"))
        throw std::runtime_error("'directory_intrinsic' is not set.");

    ros::param::get("~topicname_image", topicname_image_);
    ros::param::get("~topicname_pose_estimation", topicname_pose_estimation_);
    ros::param::get("~directory_intrinsic", directory_intrinsic_);
};

/**
 * @brief image callback function .
 * @details It is called when a new image arrives. In this function, 'scale_mono_vo->track()' function is called.
 * @param msg sensor_msgs::ImageConstPtr
 * @return none
 * @author Changhyeon Kim (hyun91015@gmail.com)
 * @date 10-July-2022
 */
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
    // scale_mono_vo_->trackImage(cv_ptr->image, cv_ptr->header.stamp.toSec());
    // Update();
    ROS_INFO_STREAM( "execution time: " << timer::toc(0) << " msec.");
};

/**
 * @brief member method including ROS spin at 200 Hz rate.
 * @details ROS spin at 200 Hz.
 * @return none
 * @author Changhyeon Kim (hyun91015@gmail.com)
 * @date 10-July-2022
 */
void MonoNode::run(){
    ros::Rate rate(1000);
    while(nh_.ok()){
        ros::spinOnce();
        rate.sleep();
    }
};
