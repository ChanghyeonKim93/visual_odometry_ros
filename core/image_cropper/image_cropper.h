#ifndef _IMAGE_CROPPER_H_
#define _IMAGE_CROPPER_H_

#include <iostream>
#include <exception>
#include <vector>
#include <string>

// ROS cv_bridge
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/image_encodings.h> 
#include <sensor_msgs/fill_image.h>
#include <image_transport/image_transport.h>

struct CropSize
{
    int n_cols;
    int n_rows;

    int n_cols_crop;
    int n_rows_crop;

    int crop_left;
    int crop_right;
    int crop_top;
    int crop_bottom;

    cv::Rect roi;
};

class ImageCropper
{
private:
    int n_cameras_;

    std::vector<std::string> topicnames_image_raw_;
    std::vector<std::string> topicnames_image_crop_;

private:
    ros::NodeHandle nh_;
    image_transport::ImageTransport          it_;
    std::vector<image_transport::Subscriber> subs_imgs_; 

    std::vector<ros::Publisher> pubs_cropped_;

    // data container (buffer)
    std::vector<cv::Mat> buf_imgs_; 
    std::vector<cv::Mat> buf_imgs_cropped_; 

    std::vector<CropSize> crop_size_;
      
public:
    ImageCropper(ros::NodeHandle& nh);
    ~ImageCropper();
    void run();

private:
    void getParameters();
    void callbackImage(const sensor_msgs::ImageConstPtr& msg, const int& id);


};

#endif