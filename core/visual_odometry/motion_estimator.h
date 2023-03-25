#ifndef _MOTION_ESTIMATOR_H_
#define _MOTION_ESTIMATOR_H_

#include <iostream>
#include <exception>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <sstream>

#include <Eigen/Dense>

#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include "core/visual_odometry/define_type.h"

#include "core/visual_odometry/camera.h"
#include "core/visual_odometry/frame.h"
#include "core/visual_odometry/keyframes.h"
#include "core/visual_odometry/landmark.h"

#include "core/visual_odometry/ba_solver/sparse_bundle_adjustment.h"

#include "core/util/histogram.h"
#include "core/util/geometry_library.h"
#include "core/util/timer.h"
#include "core/util/triangulate_3d.h"

/// @brief This class is for estimating camera motion via 2D-2D, 3D-2D feature correspondences. This class supports the 'stereo mode'.
class MotionEstimator 
{
private:
    bool is_stereo_mode_;
    PoseSE3 T_lr_;

private:
    float thres_1p_;
    float thres_5p_;

private:
    std::shared_ptr<SparseBundleAdjustmentSolver> sparse_ba_solver_;
    
public:
    MotionEstimator(bool is_stereo_mode = false,  const PoseSE3& T_lr = PoseSE3::Identity() );
    ~MotionEstimator();

    bool calcPose5PointsAlgorithm(const PixelVec& pts0, const PixelVec& pts1, CameraConstPtr& cam, 
        Rot3& R10_true, Pos3& t10_true, PointVec& X0_true, MaskVec& mask_inlier);
    bool calcPosePnPAlgorithm(const PointVec& Xw, const PixelVec& pts_c, CameraConstPtr& cam, 
        Rot3& Rwc, Pos3& twc, MaskVec& maskvec_inlier);
    float findInliers1PointHistogram(const PixelVec& pts0, const PixelVec& pts1, CameraConstPtr& cam,
        MaskVec& maskvec_inlier);

    bool poseOnlyBundleAdjustment(const PointVec& X, const PixelVec& pts1, CameraConstPtr& cam, const int& thres_reproj_outlier,
        Rot3& R01_true, Pos3& t01_true, MaskVec& mask_inlier);
    bool poseOnlyBundleAdjustment_Stereo(const PointVec& X, const PixelVec& pts_l1, const PixelVec& pts_r1, CameraConstPtr& cam_left, CameraConstPtr& cam_right, const PoseSE3& T_lr, float thres_reproj_outlier, 
        PoseSE3& T01, MaskVec& mask_inlier);

    bool localBundleAdjustmentSparseSolver(const std::shared_ptr<Keyframes>& kfs, CameraConstPtr& cam);
    bool localBundleAdjustmentSparseSolver_Stereo(const std::shared_ptr<StereoKeyframes>& stkfs_window, CameraConstPtr& cam_left, CameraConstPtr& cam_right, const PoseSE3& T_lr);

    // bool localBundleAdjustmentSparseSolver(const std::shared_ptr<Keyframes>& kfs, CameraConstPtr& cam);

public:
    void  calcSampsonDistance(const PixelVec& pts0, const PixelVec& pts1, CameraConstPtr& cam, 
                            const Rot3& R10, const Pos3& t10, std::vector<float>& sampson_dist);
    void  calcSampsonDistance(const PixelVec& pts0, const PixelVec& pts1,const Mat33& F10, 
                            std::vector<float>& sampson_dist);
    float calcSampsonDistance(const Pixel& pt0, const Pixel& pt1,const Mat33& F10);
    void  calcSymmetricEpipolarDistance(const PixelVec& pts0, const PixelVec& pts1, CameraConstPtr& cam, 
                            const Rot3& R10, const Pos3& t10, std::vector<float>& sym_epi_dist);
    
public:
    void setThres1p(float thres_1p);
    void setThres5p(float thres_5p);

private:
    bool findCorrectRT(
        const std::vector<Eigen::Matrix3f>& R10_vec, const std::vector<Eigen::Vector3f>& t10_vec, 
        const PixelVec& pxvec0, const PixelVec& pxvec1, CameraConstPtr& cam,
        Rot3& R10_true, Pos3& t10_true, 
        MaskVec& maskvec_true, PointVec& X0);

    void refineEssentialMat(const PixelVec& pts0, const PixelVec& pts1, const MaskVec& mask, CameraConstPtr& cam,
        Mat33& E);

    void refineEssentialMatIRLS(const PixelVec& pts0, const PixelVec& pts1, const MaskVec& mask, CameraConstPtr& cam,
        Mat33& E);

// Hessian related functions.
private:
    inline void calcJtJ_x(const Eigen::Matrix<float,6,1>& Jt, Eigen::Matrix<float,6,6>& JtJ_tmp);
    inline void calcJtJ_y(const Eigen::Matrix<float,6,1>& Jt, Eigen::Matrix<float,6,6>& JtJ_tmp);
    inline void calcJtWJ_x(const float weight, const Eigen::Matrix<float,6,1>& Jt, Eigen::Matrix<float,6,6>& JtJ_tmp);
    inline void calcJtWJ_y(const float weight, const Eigen::Matrix<float,6,1>& Jt, Eigen::Matrix<float,6,6>& JtJ_tmp);
};

#endif