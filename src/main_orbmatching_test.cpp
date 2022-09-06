#include <ros/ros.h>
#include <iostream>
#include <vector>

#include <time.h>
#include <string>
#include <sstream>
#include <exception>

#include <Eigen/Dense>

#include <cv_bridge/cv_bridge.h>

#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include "util/signal_handler_linux.h"

//https://roomedia.tistory.com/entry/60일차-C-openCV-4-실시간-영상에서-ORB-특징점-추출-및-FLANN-매칭하기
// custom
#include "core/feature_extractor.h"

int main(int argc, char **argv) {
    ros::init(argc, argv, "test_orb_matching_node");
    ros::NodeHandle nh("~");
    
    ROS_INFO_STREAM("test_orb_matching_node - starts.");
   
    try {  
        std::cout << "RUN!\n";
        
        // Make ORB extractor
        int n_cols = 1241;
        int n_rows = 376;
        int n_bins_u = 40;
        int n_bins_v = 24;
        int THRES_FAST = 40.0;
        int radius = 0;

        std::shared_ptr<FeatureExtractor> ext_;
        ext_ = std::make_shared<FeatureExtractor>();
        ext_->initParams(n_cols, n_rows, n_bins_u, n_bins_v, THRES_FAST, radius);

        // Load images
        std::vector<cv::Mat> imgs(5);
        imgs.at(0) = cv::imread("/home/larrkchlinux/Documents/kitti/test_orbmatching/0.png", cv::IMREAD_GRAYSCALE);
        imgs.at(1) = cv::imread("/home/larrkchlinux/Documents/kitti/test_orbmatching/1.png", cv::IMREAD_GRAYSCALE);
        imgs.at(2) = cv::imread("/home/larrkchlinux/Documents/kitti/test_orbmatching/2.png", cv::IMREAD_GRAYSCALE);
        imgs.at(3) = cv::imread("/home/larrkchlinux/Documents/kitti/test_orbmatching/3.png", cv::IMREAD_GRAYSCALE);
        imgs.at(4) = cv::imread("/home/larrkchlinux/Documents/kitti/test_orbmatching/4.png", cv::IMREAD_GRAYSCALE);

        // Extract feature and draw images
        for(int i = 0; i < 5; ++i){
            ext_->resetWeightBin();
            std::vector<cv::Point2f> pts;
            ext_->extractORBwithBinning(imgs.at(i), pts);

            cv::Mat img_draw;	
            imgs.at(i).copyTo(img_draw);
            cv::cvtColor(img_draw, img_draw, CV_GRAY2RGB);

            std::cout << "# of pixels : " << pts.size() << std::endl;
            for(int j = 0; j < pts.size(); ++j)
                cv::circle(img_draw, pts.at(j), 1.0, cv::Scalar(0,0,255),1); // alived magenta
            std::string window_name = "img" + i;
            cv::namedWindow(window_name);
            cv::imshow(window_name,img_draw);
            cv::waitKey(0);
        }
    }
    catch (std::exception& e) {
        ROS_ERROR(e.what());
    }
   
    ROS_INFO_STREAM("test_orb_matching_node - TERMINATED.");
    return -1;
}