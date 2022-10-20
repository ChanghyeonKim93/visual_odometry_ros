#ifndef _LANDMARK_BA_H_
#define _LANDMARK_BA_H_

#include <iostream>

#include "core/landmark.h"
#include "core/frame.h"
#include "core/type_defines.h"

#include "core/ba_solver/ba_types.h"

/// @brief landmark structure for a Sparse Local Bundle Adjustment (SLBA)
struct LandmarkBA
{
    _BA_Point         X; // 3D point represented in the reference frame
    FramePtrVec       kfs_seen;   // 해당 키프레임에서 어떤 좌표로 보였는지를 알아야 함.
    _BA_PixelVec      pts_on_kfs; // 각 키프레임에서 추적된 pixel 좌표.
    
    
    LandmarkPtr       lm; // 해당 landmark의 original pointer.

    /// @brief constructor of landmark structure for sparse bundle adjustment
    LandmarkBA() 
    {
        lm = nullptr;
        X  = _BA_Vec3::Zero();
    };

    /// @brief constructor of landmark structure for sparse bundle adjustment
    /// @param lmba landmark pointer of original landmark
    LandmarkBA(const LandmarkBA& lmba) 
    {
        lm              = lmba.lm;
        X               = lmba.X;
        kfs_seen        = lmba.kfs_seen;
        pts_on_kfs      = lmba.pts_on_kfs;
    };
};

#endif