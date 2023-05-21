#ifndef _FEATURE_TRACKER_H_
#define _FEATURE_TRACKER_H_

#include <iostream>
#include <vector>

// eigen
#include "eigen3/Eigen/Dense"

// opencv
#include "opencv4/opencv2/core.hpp"
#include "opencv4/opencv2/imgproc.hpp"
#include "opencv4/opencv2/highgui.hpp"
#include "opencv4/opencv2/core/eigen.hpp"
#include "opencv4/opencv2/video/tracking.hpp"

#include "core/defines/define_type.h"

#include "core/visual_odometry/landmark.h"

#include "core/util/image_processing.h"

class FeatureTracker;

class FeatureTracker{
private:

public:
    /// @brief FeatureTracker constructor
    FeatureTracker();

    /// @brief FeatureTracker destructor
    ~FeatureTracker();

    /// @brief Feature tracking (opencv KLT tracker)
    /// @param img0 previous image
    /// @param img1 current image
    /// @param pts0 pixels on previous image
    /// @param window_size KLT tracking window size
    /// @param max_pyr_lvl max pyramid level
    /// @param thres_err threshold of KLT error 
    /// @param pts_track tracked pixels
    /// @param mask_valid tracking mask
    void track(const cv::Mat& img0, const cv::Mat& img1, const PixelVec& pts0, uint32_t window_size, uint32_t max_pyr_lvl, float thres_err,
                PixelVec& pts_track, MaskVec& mask_valid);

    /// @brief Feature tracking (Bidirectional tracking)
    /// @param img0 previous image
    /// @param img1 current image
    /// @param pts0 pixels on previous image
    /// @param window_size KLT tracking window size
    /// @param max_pyr_lvl max pyramid level
    /// @param thres_err threshold of KLT error
    /// @param thres_bidirection threshold of bidirection pixel error 
    /// @param pts_track tracked pixels
    /// @param mask_valid tracking mask            
    void trackBidirection(const cv::Mat& img0, const cv::Mat& img1, const PixelVec& pts0, uint32_t window_size, uint32_t max_pyr_lvl, float thres_err, float thres_bidirection, 
                PixelVec& pts_track, MaskVec& mask_valid);

    /// @brief Feature tracking with prior pixels (Bidirectional tracking)
    /// @param img0 previous image
    /// @param img1 current image
    /// @param pts0 pixels on previous image
    /// @param window_size KLT tracking window size
    /// @param max_pyr_lvl max pyramid level
    /// @param thres_err threshold of KLT error
    /// @param thres_bidirection threshold of bidirection pixel error 
    /// @param pts_track tracked pixels
    /// @param mask_valid tracking mask
    void trackBidirectionWithPrior(const cv::Mat& img0, const cv::Mat& img1, const PixelVec& pts0, uint32_t window_size, uint32_t max_pyr_lvl, float thres_err, float thres_bidirection, 
                PixelVec& pts_track, MaskVec& mask_valid);

    /// @brief Feature tracking with prior pixels.
    /// @param img0 previous image
    /// @param img1 current image
    /// @param pts0 pixels on previous image
    /// @param window_size KLT tracking window size
    /// @param max_pyr_lvl max pyramid level
    /// @param thres_err threshold of KLT error
    /// @param pts_track tracked pixels
    /// @param mask_valid tracking mask
    void trackWithPrior(const cv::Mat& img0, const cv::Mat& img1, const PixelVec& pts0, uint32_t window_size, uint32_t max_pyr_lvl, float thres_err,
                PixelVec& pts_track, MaskVec& mask_valid);

    /// @brief Calculate tracking prior from the 3D point
    /// @param pts0 previous pixel on previous image
    /// @param Xw 3D point represented in world frame
    /// @param Tw1 SE3 from world to the current image frame
    /// @param K intrinsic matrix 
    /// @param pts1_prior prior pixel on current image
    void calcPrior(const PixelVec& pts0, const PointVec& Xw, const PoseSE3& Tw1, const Eigen::Matrix3f& K,
                PixelVec& pts1_prior);

    /// @brief Re-track the pixel with scale compensation (scale is given and fixed.)
    /// @param img0 img0 (CV_32FC1, automatically converted)
    /// @param du0 u-derivative image of img0 (CV_32FC1)
    /// @param dv0 v-derivative image of img0 (CV_32FC1)
    /// @param img1 img1 (CV_32FC1, automatically converted)
    /// @param pts0 pixels on img0
    /// @param scale_est scale from img0 to img1
    /// @param pts_track initial tracked pixels of pts0 on img1
    /// @param mask_valid mask for pts_track
    void trackWithScale(const cv::Mat& img0, const cv::Mat& du0, const cv::Mat& dv0, const cv::Mat& img1, const PixelVec& pts0, const std::vector<float>& scale_est,
                PixelVec& pts_track, MaskVec& mask_valid);

    // /// @brief refine a current pixel tracking w.r.t. the very first observation of each landmark.
    // /// @param img1 current image
    // /// @param lms landmarks
    // /// @param scale_est estimated scales from the very first observation of each landmark
    // /// @param pts_track tracking pixels on the current image (prior values)
    // /// @param mask_valid validity masks of the landmarks
    // void refineTrackWithScale(const cv::Mat& img1, const LandmarkPtrVec& lms, const std::vector<float>& scale_est,
    //             PixelVec& pts_track, MaskVec& mask_valid);
};


#endif