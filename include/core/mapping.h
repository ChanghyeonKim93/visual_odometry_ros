#ifndef _MAPPING_H_
#define _MAPPING_H_

#include <iostream>
#include <vector>

#include <Eigen/Dense>
#include "core/camera.h"

#include "core/type_defines.h"

namespace Mapping{
    void triangulateDLT(const PixelVec& pts0, const PixelVec& pts1, 
                        const Rot3& R10, const Pos3& t10, CameraConstPtr& cam, 
                        PointVec& X0, PointVec& X1);
    
    void triangulateDLT(const Pixel& pt0, const Pixel& pt1, 
                        const Rot3& R10, const Pos3& t10, CameraConstPtr& cam, 
                        Point& X0, Point& X1);

    void triangulateDLT(const Pixel& pt0, const Pixel& pt1, 
                        const Rot3& R10, const Pos3& t10, CameraConstPtr& cam0, CameraConstPtr& cam1, 
                        Point& X0, Point& X1);
// TODO
    void calcStereoDisparity_RectifiedStatic(const PixelVec& pts0, CameraConstPtr& cam_rect);

    void a();

    Eigen::Matrix3f skew(const Eigen::Vector3f& vec);
};

#endif