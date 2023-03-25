#ifndef _LANDMARK_BA_H_
#define _LANDMARK_BA_H_

#include <iostream>

#include "core/visual_odometry/landmark.h"
#include "core/visual_odometry/frame.h"

#include "core/visual_odometry/ba_solver/define_ba_type.h"

/// @brief landmark structure for a Sparse Local Bundle Adjustment (SLBA)
struct LandmarkBA {
    _BA_Point    X; // 3D point represented in the reference frame
    FramePtrVec  kfs_seen;   // 해당 키프레임에서 어떤 좌표로 보였는지를 알아야 함.
    _BA_PixelVec pts_on_kfs; // 각 키프레임에서 추적된 pixel 좌표.
    _BA_ErrorVec err_on_kfs; // 각 키프레임에서의 reprojection error.

    LandmarkPtr  lm; // 해당 landmark의 original pointer.

    /// @brief constructor of landmark structure for sparse bundle adjustment
    LandmarkBA() 
    {
        lm = nullptr;
        X  = _BA_Vec3::Zero();
        kfs_seen.reserve(300);
        pts_on_kfs.reserve(300);
        err_on_kfs.reserve(300);
    };

    /// @brief constructor of landmark structure for sparse bundle adjustment
    /// @param lmba landmark pointer of original landmark
    LandmarkBA(const LandmarkBA& lmba) 
    {
        lm              = lmba.lm;
        X               = lmba.X;
        kfs_seen        = lmba.kfs_seen;
        pts_on_kfs      = lmba.pts_on_kfs;
        err_on_kfs      = lmba.err_on_kfs;
    };

public:
    void setErrorByIndex(_BA_Numeric err, int jj){
        err_on_kfs.at(jj) = err;
    };
};

#endif