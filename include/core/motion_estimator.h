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

class MotionEstimator;

class MotionEstimator{
public:
    MotionEstimator();
    ~MotionEstimator();

    bool calcPose5PointsAlgorithm(const PixelVec& pts0, const PixelVec& pts1, const std::shared_ptr<Camera>& cam, 
        MaskVec& mask_inlier);
    bool calcPosePnPAlgorithm(const PointVec& Xw, const PixelVec& pts1);

private:

    bool verifySolution(const std::vector<Eigen::Matrix3f>& R_vec,
                        const std::vector<Eigen::Vector3f>& t_vec, 
                        Eigen::Matrix3f& R, 
                        Eigen::Vector3f& t, 
                        MaskVec& max_inlier, 
                        PointVec& opt_X_curr);

    void triangulate(const PixelVec& pts0, const PixelVec& pts1, 
                     const Eigen::Matrix3f& R10, const Eigen::Vector3f& t10, 
                     PointVec& X0, PointVec& X1,
                     MaskVec& inlier);
};

#endif