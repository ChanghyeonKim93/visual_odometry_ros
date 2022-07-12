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

class MotionEstimator;

class MotionEstimator{
public:
    MotionEstimator();
    ~MotionEstimator();

    bool calcPose5PointsAlgorithm(const PixelVec& pts0, const PixelVec& pts1, const std::shared_ptr<Camera>& cam, 
        Eigen::Matrix3f& R10_true, Eigen::Vector3f& t10_true, MaskVec& mask_inlier);
    bool calcPosePnPAlgorithm(const PointVec& Xw, const PixelVec& pts1);

private:
 
    bool findCorrectRT(
        const std::vector<Eigen::Matrix3f>& R10_vec, const std::vector<Eigen::Vector3f>& t10_vec, 
        const PixelVec& pxvec0, const PixelVec& pxvec1, const std::shared_ptr<Camera>& cam,
        Eigen::Matrix3f& R10_true, Eigen::Vector3f& t10_true, 
        MaskVec& max_inlier, PointVec& X0);
};

#endif