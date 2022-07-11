#ifndef _FEATURE_TRACKER_H_
#define _FEATURE_TRACKER_H_

#include <iostream>
#include <vector>

// ROS eigen
#include <Eigen/Dense>

// ROS cv_bridge
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/eigen.hpp>

#include "core/type_defines.h"

using namespace std;
class FeatureTracker;

class FeatureTracker{
private:

public:
    FeatureTracker();
    ~FeatureTracker();

    void track(const cv::Mat& img0, const cv::Mat& img1, const PixelVec& pts0, 
                PixelVec& pts_track, MaskVec& mask_valid);
    void trackBidirection(const cv::Mat& img0, const cv::Mat& img1, const PixelVec& pts0, float thres_err, float thres_bidirection, 
                PixelVec& pts_track, MaskVec& mask_valid);
    void trackWithPrior(const cv::Mat& img0, const cv::Mat& img1, const PixelVec& pts0, const PixelVec& pts1_prior,
                PixelVec& pts_track, MaskVec& mask_valid);
    void calcPrior(const PixelVec& pts0, const PointVec& Xw, const Eigen::Matrix4f& Tw1, const Eigen::Matrix3f& K,
                PixelVec& pts1_prior);
};


#endif