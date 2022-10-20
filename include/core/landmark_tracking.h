#ifndef _LANDMARK_TRACKING_H_
#define _LANDMARK_TRACKING_H_

#include <iostream>
#include <vector>
#include <memory>
#include <Eigen/Dense>

#include "core/type_defines.h"

#include "core/frame.h"

/// @brief A temporal structur for monocular feature tracking 
class LandmarkTracking
{
private:
    PixelVec pts0_;
    PixelVec pts1_;
    LandmarkPtrVec lms_;
    std::vector<float> scale_change_;

    int n_pts_;
    
public:
    LandmarkTracking();
    LandmarkTracking(const LandmarkTracking& lmtrack, const MaskVec& mask);

    void initializeFromFramePtr(const FramePtr& f)
    {
        const PixelVec&      pts0_frame = f->getPtsSeen();
        const LandmarkPtrVec& lms_frame = f->getRelatedLandmarkPtr();

        int n_pts_frame = pts0_frame.size();
        
        pts0.reserve(n_pts_frame);
        lms.reserve(n_pts_frame);
        pts1.reserve(n_pts_frame); 
        scale_change.reserve(n_pts_frame);

        // Get only valid points
        for(int i = 0; i < n_pts_frame; ++i)
        {
            const LandmarkPtr& lm = lms_frame[i];

            if( lm->isAlive() )
            {
                pts0.emplace_back(pts0_frame[i]);
                lms.push_back(lm);
            }
        }

        pts1.resize(pts0.size());
        scale_change.resize(pts0.size());
        
        n_pts = pts1.size();
    };
};

#endif