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
    img_sub_ = nh_.subscribe<sensor_msgs::Image>(topicname_image_, 1, &MonoNode::imageCallback, this);

    // Publisher
    pub_pose_       = nh_.advertise<nav_msgs::Odometry>(topicname_pose_, 1);
    pub_trajectory_ = nh_.advertise<nav_msgs::Path>(topicname_trajectory_, 1);
    pub_map_points_ = nh_.advertise<sensor_msgs::PointCloud2>(topicname_map_points_, 1);

    topicname_statistics_ = "/scale_mono_vo/statistics";
    pub_statistics_ = nh_.advertise<scale_mono_vo_ros::statisticsStamped>(topicname_statistics_,1);

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
    if(!ros::param::has("~topicname_pose"))
        throw std::runtime_error("'topicname_pose' is not set.");
    if(!ros::param::has("~topicname_map_points"))
        throw std::runtime_error("'topicname_map_points' is not set.");
    if(!ros::param::has("~topicname_trajectory"))
        throw std::runtime_error("'topicname_trajectory' is not set.");
    if(!ros::param::has("~directory_intrinsic"))
        throw std::runtime_error("'directory_intrinsic' is not set.");

    ros::param::get("~topicname_image", topicname_image_);

    ros::param::get("~topicname_pose",       topicname_pose_);
    ros::param::get("~topicname_trajectory", topicname_trajectory_);
    ros::param::get("~topicname_map_points", topicname_map_points_);
    
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
        throw std::runtime_error("ERR!!!!");
        return;
    }    
    // update camera pose.
    double time_now = cv_ptr->header.stamp.toSec();
    scale_mono_vo_->trackImage(cv_ptr->image, time_now);

    // Get odometry results
    ScaleMonoVO::AlgorithmStatistics stat;
    stat = scale_mono_vo_->getStatistics();
    
    std::cout << "====================================================\n";
    std::cout << "----------------------------\n";
    std::cout << "total time: " << stat.stats_execution.back().time_total <<" ms\n";
    std::cout << "     track: " << stat.stats_execution.back().time_track <<" ms\n";
    std::cout << "        1p: " << stat.stats_execution.back().time_1p <<" ms\n";
    std::cout << "        5p: " << stat.stats_execution.back().time_5p <<" ms\n";
    std::cout << "       new: " << stat.stats_execution.back().time_new <<" ms\n";

    std::cout << "----------------------------\n";
    std::cout << "landmark init.: " << stat.stats_landmark.back().n_initial <<"\n";
    std::cout << "         track: " << stat.stats_landmark.back().n_pass_bidirection <<"\n";
    std::cout << "            1p: " << stat.stats_landmark.back().n_pass_1p <<"\n";
    std::cout << "            5p: " << stat.stats_landmark.back().n_pass_5p <<"\n";
    std::cout << "           new: " << stat.stats_landmark.back().n_new <<"\n";
    std::cout << "         final: " << stat.stats_landmark.back().n_final <<"\n";
    std::cout << "   parallax ok: " << stat.stats_landmark.back().n_ok_parallax <<"\n\n";

    std::cout << " avg. parallax: " << stat.stats_landmark.back().avg_parallax <<" rad \n";
    std::cout << " avg.      age: " << stat.stats_landmark.back().avg_age <<" frames \n";

    std::cout << "----------------------------\n";
    std::cout << "      steering: " << stat.stats_frame.back().steering_angle << " rad\n";
    std::cout << "====================================================\n";

    scale_mono_vo_ros::statisticsStamped msg_statistics;
    
    msg_statistics.time_total = stat.stats_execution.back().time_total; 
    msg_statistics.time_track = stat.stats_execution.back().time_track; 
    msg_statistics.time_1p = stat.stats_execution.back().time_1p;    
    msg_statistics.time_5p = stat.stats_execution.back().time_5p;
    msg_statistics.time_new = stat.stats_execution.back().time_new;

    msg_statistics.n_initial = stat.stats_landmark.back().n_initial;
    msg_statistics.n_pass_bidirection = stat.stats_landmark.back().n_pass_bidirection;
    msg_statistics.n_pass_1p = stat.stats_landmark.back().n_pass_1p;
    msg_statistics.n_pass_5p = stat.stats_landmark.back().n_pass_5p;
    msg_statistics.n_new = stat.stats_landmark.back().n_new;
    msg_statistics.n_final = stat.stats_landmark.back().n_final;
    msg_statistics.n_ok_parallax = stat.stats_landmark.back().n_ok_parallax;

    msg_statistics.avg_parallax = stat.stats_landmark.back().avg_parallax;
    msg_statistics.avg_age = stat.stats_landmark.back().avg_age;

    msg_statistics.steering_angle = stat.stats_frame.back().steering_angle;
    pub_statistics_.publish(msg_statistics);

    // Pose publish
    nav_msgs::Odometry msg_pose;
    msg_pose.header.stamp = ros::Time::now();
    msg_pose.header.frame_id = "map";

    PoseSE3 Twc = stat.stats_frame.back().Twc;
    msg_pose.pose.pose.position.x = Twc(0,3);
    msg_pose.pose.pose.position.y = Twc(1,3);
    msg_pose.pose.pose.position.z = Twc(2,3);
    
    Eigen::Vector4f q = geometry::r2q_f(Twc.block<3,3>(0,0));

    msg_pose.pose.pose.orientation.w = q(0);
    msg_pose.pose.pose.orientation.x = q(1);
    msg_pose.pose.pose.orientation.y = q(2);
    msg_pose.pose.pose.orientation.z = q(3);

    pub_pose_.publish(msg_pose);

    geometry_msgs::PoseStamped p;
    p.header.frame_id = "map";
    p.header.stamp = ros::Time::now();
    p.pose = msg_pose.pose.pose;

    msg_trajectory_.header.frame_id = "map";
    msg_trajectory_.header.stamp = ros::Time::now();
    msg_trajectory_.poses.push_back(p);

    pub_trajectory_.publish(msg_trajectory_);
};

/**
 * @brief member method including ROS spin at 200 Hz rate.
 * @details ROS spin at 200 Hz.
 * @return none
 * @author Changhyeon Kim (hyun91015@gmail.com)
 * @date 10-July-2022
 */
void MonoNode::run(){
    ros::Rate rate(200);
    while(ros::ok()){
        ros::spinOnce();
        rate.sleep();
    }
};
