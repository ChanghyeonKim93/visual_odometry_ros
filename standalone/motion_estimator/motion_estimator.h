#ifndef _MOTION_ESTIMATOR_H_
#define _MOTION_ESTIMATOR_H_

#include <iostream>
#include <exception>
#include <vector>

#include "eigen3/Eigen/Dense"
#include "define_type.h"

#include "histogram.h"
#include "geometry_library.h"
#include "timer.h"
#include "triangulate_3d.h"

class MotionEstimator
{
private:
  bool is_stereo_mode_;
  PoseSE3 T_left2right_;

public:
  MotionEstimator(bool is_stereo_mode = false, const PoseSE3 &T_left2right = PoseSE3::Identity());
  ~MotionEstimator();

  // bool poseOnlyBundleAdjustment(const PointVec &X, const PixelVec &pts1, CameraConstPtr &cam, const int &thres_reproj_outlier,
  //                               Rot3 &R01_true, Pos3 &t01_true, MaskVec &mask_inlier);
  // bool poseOnlyBundleAdjustment_Stereo(const PointVec &X, const PixelVec &pts_l1, const PixelVec &pts_r1, CameraConstPtr &cam_left, CameraConstPtr &cam_right, const PoseSE3 &T_lr, float thres_reproj_outlier,
  //                                      PoseSE3 &T01, MaskVec &mask_inlier);

  bool poseOnlyBundleAdjustment(const PointVec &X, const PixelVec &pts1, const float fx, const float fy, const float cx, const float cy, const int &thres_reproj_outlier,
                                Rot3 &R01_true, Pos3 &t01_true, MaskVec &mask_inlier);
  bool poseOnlyBundleAdjustment_Stereo(const PointVec &X, const PixelVec &pts_l1, const PixelVec &pts_r1,
                                       const float fx_l, const float fy_l, const float cx_l, const float cy_l, const float fx_r, const float fy_r, const float cx_r, const float cy_r, 
                                       const PoseSE3 &T_lr, float thres_reproj_outlier,
                                       PoseSE3 &T01, MaskVec &mask_inlier);

  // Hessian related functions.
private:
  inline void calcJtJ_x(const Vec6 &Jt, Mat66 &JtJ_tmp);
  inline void calcJtJ_y(const Vec6 &Jt, Mat66 &JtJ_tmp);
  inline void calcJtWJ_x(const float weight, const Vec6 &Jt, Mat66 &JtJ_tmp);
  inline void calcJtWJ_y(const float weight, const Vec6 &Jt, Mat66 &JtJ_tmp);
};
#endif