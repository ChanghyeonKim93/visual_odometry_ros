#ifndef _TRIANGULATE_3D_H_
#define _TRIANGULATE_3D_H_

#include <iostream>
#include <vector>

// #include <eigen3/dense>
#include "eigen3/Eigen/Dense"

#include "motion_estimator/define_type.h"

namespace mapping
{
    void triangulateDLT(const PixelVec& pts0, const PixelVec& pts1, 
                        const Rot3& R10, const Pos3& t10, 
                        const float fx, const float fy, const float cx, const float cy,
                        PointVec& X0, PointVec& X1);
    
    void triangulateDLT(const Pixel& pt0, const Pixel& pt1, 
                        const Rot3& R10, const Pos3& t10, 
                        const float fx, const float fy, const float cx, const float cy,
                        Point& X0, Point& X1);

    void triangulateDLT(const Pixel& pt0, const Pixel& pt1, 
                        const Rot3& R10, const Pos3& t10, 
                        const float fx_l, const float fy_l, const float cx_l, const float cy_l,
                        const float fx_r, const float fy_r, const float cx_r, const float cy_r,
                        Point& X0, Point& X1);

    Eigen::Matrix3f skew(const Vec3& vec);
};

#endif