#ifndef _BACKEND_NODE_H_
#define _BACKEND_NODE_H_

#include <iostream>
#include <algorithm>
#include <fstream>

#include <Eigen/Dense>

#include <ros/ros.h>
#include <std_msgs/Float32.h>
#include <sensor_msgs/Image.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/PointCloud2.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>

#include "core/scale_mono_vo/scale_mono_vo.h"

#include "util/timer.h"

#include "scale_mono_vo_ros/statisticsStamped.h"
#include "util/geometry_library.h"

class BackendNode{
public:
    BackendNode(ros::NodeHandle& nh);
    ~BackendNode();

private:
    void getParameters();
    void doTracking(const cv::Mat& img, const PoseSE3& pose, const PoseSE3& dT01);
    void run();

private:
    void imageFromExternalVOCallback(const sensor_msgs::ImageConstPtr& msg);
    void poseFromExternalVOCallback(const geometry_msgs::PoseStampedConstPtr& msg);

    void groundtruthCallback(const geometry_msgs::PoseStampedConstPtr& msg);
 
private:
    void convertPointVecToPointCloud2(const PointVec& X, sensor_msgs::PointCloud2& dst, std::string frame_id);

private:
    ros::NodeHandle nh_;
    
    // subscriber
    ros::Subscriber img_sub_;
    std::string topicname_image_from_external_vo_;

    ros::Subscriber pose_sub_;
    std::string topicname_pose_from_external_vo_;

    bool flag_image_got_;
    bool flag_pose_got_;

    double time_img_cur_;
    double time_pose_cur_;
    cv::Mat img_cur_;
    PoseSE3 pose_cur_;
    PoseSE3 pose_prev_;

    ros::Subscriber gt_sub_;
    std::string topicname_gt_;

    // publishers
    ros::Publisher pub_pose_;
    std::string topicname_pose_;

    ros::Publisher pub_trajectory_;
    nav_msgs::Path msg_trajectory_;
    std::string topicname_trajectory_;

    ros::Publisher pub_map_points_;
    PointVec mappoints_;
    std::string topicname_map_points_;

    ros::Publisher pub_statistics_;
    std::string topicname_statistics_;

    // Publishers for turn region detections
    ros::Publisher pub_turns_;
    sensor_msgs::PointCloud2 msg_turns_;
    std::string topicname_turns_;

    // Publishers for ground truth
    ros::Publisher pub_trajectory_gt_;
    nav_msgs::Path msg_trajectory_gt_;
    std::string topicname_trajectory_gt_;


    // Publishers for scales
    Vec3 trans_prev_gt_;
    Vec3 trans_curr_gt_;
    float scale_gt_;
    
private:
    std::string directory_intrinsic_;

private:
    std::unique_ptr<ScaleMonoVO> scale_mono_vo_;
};

#endif