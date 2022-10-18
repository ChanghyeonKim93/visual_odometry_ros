#include "ros_wrapper/mononode.h"

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
    gt_sub_  = nh_.subscribe<geometry_msgs::PoseStamped>(topicname_gt_, 1, &MonoNode::groundtruthCallback, this);

    // Publisher
    pub_pose_       = nh_.advertise<nav_msgs::Odometry>(topicname_pose_, 1);
    pub_trajectory_ = nh_.advertise<nav_msgs::Path>(topicname_trajectory_, 1);
    pub_map_points_ = nh_.advertise<sensor_msgs::PointCloud2>(topicname_map_points_, 1);

    topicname_turns_ = "/kitti_odometry/turns";
    pub_turns_ = nh_.advertise<sensor_msgs::PointCloud2>(topicname_turns_,1);

    topicname_trajectory_gt_ = "/kitti_odometry/groundtruth_path";
    pub_trajectory_gt_ = nh_.advertise<nav_msgs::Path>(topicname_trajectory_gt_,1);

    topicname_statistics_ = "/scale_mono_vo/statistics";
    pub_statistics_ = nh_.advertise<scale_mono_vo_ros::statisticsStamped>(topicname_statistics_,1);

    pub_debug_image_ = nh_.advertise<sensor_msgs::Image>("/scale_mono_vo/debug_image",1);

    // scale publisher
    trans_prev_gt_ << 0,0,0;
    scale_gt_ = 0;
        
    ROS_INFO_STREAM("MonoNode - generate Scale Mono VO object. Starts.");

    // Set static
    int half_win_sz = 7;
    Landmark::setPatch(half_win_sz);

    mappoints_.reserve(500000);

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
    if(!ros::param::has("~topicname_gt"))
        throw std::runtime_error("'topicname_gt' is not set.");
    if(!ros::param::has("~topicname_pose"))
        throw std::runtime_error("'topicname_pose' is not set.");
    if(!ros::param::has("~topicname_map_points"))
        throw std::runtime_error("'topicname_map_points' is not set.");
    if(!ros::param::has("~topicname_trajectory"))
        throw std::runtime_error("'topicname_trajectory' is not set.");
    if(!ros::param::has("~directory_intrinsic"))
        throw std::runtime_error("'directory_intrinsic' is not set.");

    ros::param::get("~topicname_image", topicname_image_);
    ros::param::get("~topicname_gt", topicname_gt_);

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

    ros::Time t_callback_start = ros::Time::now();

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
    ros::Time t_track_start = ros::Time::now();
    double time_now = cv_ptr->header.stamp.toSec();
    scale_mono_vo_->trackImage(cv_ptr->image, time_now);
    ros::Time t_track_end = ros::Time::now();

    ROS_GREEN_STREAM("Time for track: " << (t_track_end.toSec() - t_track_start.toSec()) * 1000.0 << " [ms]");

    
    // Show statistics & get odometry results

    const ScaleMonoVO::AlgorithmStatistics& stat = scale_mono_vo_->getStatistics();
    
    scale_mono_vo_ros::statisticsStamped msg_stats;
    msg_stats.time_total         = stat.stats_execution.back().time_total; 
    msg_stats.time_track         = stat.stats_execution.back().time_track; 
    msg_stats.time_1p            = stat.stats_execution.back().time_1p;    
    msg_stats.time_5p            = stat.stats_execution.back().time_5p;
    msg_stats.time_new           = stat.stats_execution.back().time_new;


    msg_stats.n_initial          = stat.stats_landmark.back().n_initial;
    msg_stats.n_pass_bidirection = stat.stats_landmark.back().n_pass_bidirection;
    msg_stats.n_pass_1p          = stat.stats_landmark.back().n_pass_1p;
    msg_stats.n_pass_5p          = stat.stats_landmark.back().n_pass_5p;
    msg_stats.n_new              = stat.stats_landmark.back().n_new;
    msg_stats.n_final            = stat.stats_landmark.back().n_final;
    msg_stats.n_ok_parallax      = stat.stats_landmark.back().n_ok_parallax;


    msg_stats.avg_parallax       = stat.stats_landmark.back().avg_parallax;
    msg_stats.avg_age            = stat.stats_landmark.back().avg_age;


    msg_stats.steering_angle     = stat.stats_frame.back().steering_angle;


    msg_stats.scale_est          = stat.stats_frame.back().dT_01.block<3,1>(0,3).norm();
    msg_stats.scale_gt           = scale_gt_;

    pub_statistics_.publish(msg_stats);



    ros::Time t_stat_start = ros::Time::now();

    // Pose publish
    nav_msgs::Odometry msg_pose;
    msg_pose.header.stamp = ros::Time::now();
    msg_pose.header.frame_id = "map";

    const PoseSE3& Twc = stat.stats_frame.back().Twc;
    Eigen::Vector4f q = geometry::r2q_f(Twc.block<3,3>(0,0));
    msg_pose.pose.pose.position.x = Twc(0,3);
    msg_pose.pose.pose.position.y = Twc(1,3);
    msg_pose.pose.pose.position.z = Twc(2,3);
    msg_pose.pose.pose.orientation.w = q(0);
    msg_pose.pose.pose.orientation.x = q(1);
    msg_pose.pose.pose.orientation.y = q(2);
    msg_pose.pose.pose.orientation.z = q(3);

    pub_pose_.publish(msg_pose);


    // Publish path
    msg_trajectory_.header.frame_id = "map";
    msg_trajectory_.header.stamp = ros::Time::now();

    geometry_msgs::PoseStamped p;
    p.header.frame_id = "map";
    p.header.stamp = ros::Time::now();
    p.pose = msg_pose.pose.pose;
    msg_trajectory_.poses.push_back(p);

    msg_trajectory_.poses.resize(stat.stats_keyframe.size());
    for(int j = 0; j < msg_trajectory_.poses.size(); ++j)
    {
        PoseSE3 Twc = stat.stats_keyframe[j].Twc;
        msg_pose.pose.pose.position.x = Twc(0,3);
        msg_pose.pose.pose.position.y = Twc(1,3);
        msg_pose.pose.pose.position.z = Twc(2,3);
        msg_trajectory_.poses[j].pose = msg_pose.pose.pose;
    }

    pub_trajectory_.publish(msg_trajectory_);


    // Publish mappoints
    // sensor_msgs::PointCloud2 msg_mappoint;
    // int cnt_total_pts = 0;
    // for(int j = 0; j < stat.stats_keyframe.size(); ++j)
    // {
    //     cnt_total_pts += stat.stats_keyframe[j].mappoints.size();
    // }
    // mappoints_.resize(cnt_total_pts);
    // cnt_total_pts = 0;
    // for(int j = 0; j < stat.stats_keyframe.size(); ++j){
    //     for(const auto& x : stat.stats_keyframe[j].mappoints){
    //         mappoints_[cnt_total_pts] = x;
    //         ++cnt_total_pts;
    //     }
    // }
    // convertPointVecToPointCloud2(mappoints_,msg_mappoint, "map");
    // pub_map_points_.publish(msg_mappoint);
    
    ros::Time t_stat_end = ros::Time::now();

    ROS_GREEN_STREAM("Time for STATISTICS: " << (t_stat_end.toSec() - t_stat_start.toSec()) * 1000.0 << " [ms]\n");   



    // Turn region display
    // msg_turns_.header.frame_id = "map";
    // msg_turns_.header.stamp = ros::Time::now();
    // PointVec X;
    // for(auto f : stat.stat_turn.turn_regions){
    //     X.push_back(f->getPose().block<3,1>(0,3));
    // }
    // convertPointVecToPointCloud2(X,msg_turns_, "map");
    // pub_turns_.publish(msg_turns_);


    // Debug image
    
    const cv::Mat& img_debug = scale_mono_vo_->getDebugImage();
    cv_bridge::CvImage reduced_msg;
    reduced_msg.header.stamp = ros::Time::now(); // Same timestamp and tf frame as input image
    reduced_msg.header.frame_id = "debug_image";
    reduced_msg.encoding        = "bgr8"; // Or whatever
    reduced_msg.image           = img_debug; // Your cv::Mat

    pub_debug_image_.publish(reduced_msg.toImageMsg());



    ros::Time t_callback_end = ros::Time::now();

    ROS_GREEN_STREAM("Time for CALLBACK: " << (t_callback_end.toSec() - t_callback_start.toSec()) * 1000.0 << " [ms]\n");   
};

void MonoNode::groundtruthCallback(const geometry_msgs::PoseStampedConstPtr& msg){
    geometry_msgs::PoseStamped p;
    p.header.frame_id = "map";
    p.header.stamp = ros::Time::now();
    p.pose = msg->pose;
    trans_curr_gt_ << p.pose.position.x, p.pose.position.y, p.pose.position.z;
    scale_gt_ = (trans_curr_gt_-trans_prev_gt_).norm();
    trans_prev_gt_ = trans_curr_gt_;

    msg_trajectory_gt_.header.frame_id = "map";
    msg_trajectory_gt_.header.stamp = ros::Time::now();
    msg_trajectory_gt_.poses.push_back(p);

    pub_trajectory_gt_.publish(msg_trajectory_gt_);
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

void MonoNode::convertPointVecToPointCloud2(const PointVec& X, sensor_msgs::PointCloud2& dst, std::string frame_id){
    int n_pts = X.size();
    
    // intensity mapping (-3 m ~ 3 m to 0~255)
    float z_min = -3.0;
    float z_max = 3.0;
    float intensity_min = 30;
    float intensity_max = 255;
    float slope = (intensity_max-intensity_min)/(z_max-z_min);

    dst.header.frame_id = frame_id;
    dst.header.stamp    = ros::Time::now();
    // ROS_INFO_STREAM(dst.header.stamp << endl);
    dst.width            = n_pts;
    dst.height           = 1;

    sensor_msgs::PointField f_tmp;
    f_tmp.offset = 0;    f_tmp.name="x"; f_tmp.datatype = sensor_msgs::PointField::FLOAT32; dst.fields.push_back(f_tmp);
    f_tmp.offset = 4;    f_tmp.name="y"; f_tmp.datatype = sensor_msgs::PointField::FLOAT32; dst.fields.push_back(f_tmp);
    f_tmp.offset = 8;    f_tmp.name="z"; f_tmp.datatype = sensor_msgs::PointField::FLOAT32; dst.fields.push_back(f_tmp);
    f_tmp.offset = 12;   f_tmp.name="intensity"; f_tmp.datatype = sensor_msgs::PointField::FLOAT32;  dst.fields.push_back(f_tmp);
    f_tmp.offset = 16;   f_tmp.name="ring"; f_tmp.datatype = sensor_msgs::PointField::UINT16;  dst.fields.push_back(f_tmp);
    f_tmp.offset = 18;   f_tmp.name="time"; f_tmp.datatype = sensor_msgs::PointField::FLOAT32; dst.fields.push_back(f_tmp);
    dst.point_step = 22; // x 4 + y 4 + z 4 + i 4 + r 2 + t 4 

    dst.data.resize(dst.point_step * dst.width);
    for(int i = 0; i < dst.width; ++i){
        int i_ptstep = i*dst.point_step;
        int arrayPosX = i_ptstep + dst.fields[0].offset; // X has an offset of 0
        int arrayPosY = i_ptstep + dst.fields[1].offset; // Y has an offset of 4
        int arrayPosZ = i_ptstep + dst.fields[2].offset; // Z has an offset of 8

        int ind_intensity = i_ptstep + dst.fields[3].offset; // 12
        int ind_ring      = i_ptstep + dst.fields[4].offset; // 16
        int ind_time      = i_ptstep + dst.fields[5].offset; // 18

        float height_intensity = slope*(X[i](2)-z_min)+intensity_min;
        if(height_intensity >= intensity_max) height_intensity = intensity_max;
        if(height_intensity <= intensity_min) height_intensity = intensity_min;

        float x = X[i](0);
        float y = X[i](1);
        float z = X[i](2);
        
        memcpy(&dst.data[arrayPosX],     &(x),          sizeof(float));
        memcpy(&dst.data[arrayPosY],     &(y),          sizeof(float));
        memcpy(&dst.data[arrayPosZ],     &(z),          sizeof(float));
        memcpy(&dst.data[ind_intensity], &(height_intensity),  sizeof(float));
        memcpy(&dst.data[ind_ring],      &(x),          sizeof(unsigned short));
        memcpy(&dst.data[ind_time],      &(x),          sizeof(float));
    }
};