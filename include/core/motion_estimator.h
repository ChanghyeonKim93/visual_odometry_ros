#ifndef _MOTION_ESTIMATOR_H_
#define _MOTION_ESTIMATOR_H_

#include <iostream>
#include <exception>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include "core/type_defines.h"

#include "core/camera.h"
#include "core/mapping.h"

#include "util/histogram.h"

class MotionEstimator;

class MotionEstimator{
public:
    MotionEstimator();
    ~MotionEstimator();

    bool calcPose5PointsAlgorithm(const PixelVec& pts0, const PixelVec& pts1, const std::shared_ptr<Camera>& cam, 
        Rot3& R10_true, Pos3& t10_true, PointVec& X0_true, MaskVec& mask_inlier);
    bool calcPosePnPAlgorithm(const PointVec& Xw, const PixelVec& pts_c, const std::shared_ptr<Camera>& cam, 
        Rot3& Rwc, Pos3& twc, MaskVec& maskvec_inlier);
    bool fineInliers1PointHistogram(const PixelVec& pts0, const PixelVec& pts1, const std::shared_ptr<Camera>& cam,
        MaskVec& maskvec_inlier);

    void calcSampsonDistance(const PixelVec& pts0, const PixelVec& pts1, const std::shared_ptr<Camera>& cam, 
                            const Rot3& R10, const Pos3& t10, std::vector<float>& sampson_dist);
private:
 
    bool findCorrectRT(
        const std::vector<Eigen::Matrix3f>& R10_vec, const std::vector<Eigen::Vector3f>& t10_vec, 
        const PixelVec& pxvec0, const PixelVec& pxvec1, const std::shared_ptr<Camera>& cam,
        Rot3& R10_true, Pos3& t10_true, 
        MaskVec& maskvec_true, PointVec& X0);
};

#endif