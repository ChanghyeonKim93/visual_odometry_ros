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

#include "core/landmark.h"
#include "core/type_defines.h"
#include "core/image_processing.h"

using namespace std;
class FeatureTracker;

class FeatureTracker{
private:

public:
    FeatureTracker();
    ~FeatureTracker();

    void track(const cv::Mat& img0, const cv::Mat& img1, const PixelVec& pts0, uint32_t window_size, uint32_t max_pyr_lvl, float thres_err,
                PixelVec& pts_track, MaskVec& mask_valid);
    void scaleRefinement(const cv::Mat& img0, const cv::Mat& img1, const PixelVec& pts0,  uint32_t window_size, float thres_err,
                PixelVec& pts_track, MaskVec& mask_valid);
    void trackBidirection(const cv::Mat& img0, const cv::Mat& img1, const PixelVec& pts0, uint32_t window_size, uint32_t max_pyr_lvl, float thres_err, float thres_bidirection, 
                PixelVec& pts_track, MaskVec& mask_valid);
    void trackBidirectionWithPrior(const cv::Mat& img0, const cv::Mat& img1, const PixelVec& pts0, uint32_t window_size, uint32_t max_pyr_lvl, float thres_err, float thres_bidirection, 
                PixelVec& pts_track, MaskVec& mask_valid);
    void trackWithPrior(const cv::Mat& img0, const cv::Mat& img1, const PixelVec& pts0, uint32_t window_size, uint32_t max_pyr_lvl, float thres_err,
                PixelVec& pts_track, MaskVec& mask_valid);
    void calcPrior(const PixelVec& pts0, const PointVec& Xw, const PoseSE3& Tw1, const Eigen::Matrix3f& K,
                PixelVec& pts1_prior);

    void trackWithScale(const cv::Mat& img0, const cv::Mat& img1, const cv::Mat& dimg0_u, const cv::Mat& dimg0_v, const PixelVec& pts0, const std::vector<float>& scale_est,
                PixelVec& pts_track, MaskVec& mask_valid);

    void refineTrackWithScale(const cv::Mat& img1, const LandmarkPtrVec& lms, const std::vector<float>& scale_est,
                PixelVec& pts_track, MaskVec& mask_valid);
};


#endif