#ifndef _MONO_VO_NODE_H_
#define _MONO_VO_NODE_H_

#include <iostream>
#include <algorithm>
#include <fstream>

#include <Eigen/Dense>

#include <ros/ros.h>
#include <std_msgs/Float32.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/PointCloud2.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>

#include <image_transport/image_transport.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>

#include <message_filters/sync_policies/approximate_time.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>

// My custom code
#include "core/mono_vo/mono_vo.h"

#include "util/timer.h"

#include "visual_odometry_ros/statisticsStamped.h"
#include "util/geometry_library.h"

#include "ros_wrapper/ros_print_in_color.h"

class MonoVONode{
public:
    MonoVONode(ros::NodeHandle& nh);
    ~MonoVONode();

private:
    void getParameters();

private:
    void imageCallback(const sensor_msgs::ImageConstPtr& msg);
    void groundtruthCallback(const geometry_msgs::PoseStampedConstPtr& msg);
    void run();

private:
    void convertPointVecToPointCloud2(const PointVec& X, sensor_msgs::PointCloud2& dst, std::string frame_id);

private:
    ros::NodeHandle nh_;
    
    // subscriber
    ros::Subscriber img_sub_;
    std::string topicname_image_;

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

    // Publisher for debug image
    ros::Publisher pub_debug_image_;
    sensor_msgs::Image msg_debug_image_;
    


    // Publishers for scales
    Vec3 trans_prev_gt_;
    Vec3 trans_curr_gt_;
    float scale_gt_;
    
private:
    std::string directory_intrinsic_;

private:
    std::unique_ptr<MonoVO> mono_vo_;
};

#endif