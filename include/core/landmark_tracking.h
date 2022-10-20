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
    
public:
    void initializeFromFramePtr(const FramePtr& f);

public:
    int getNumLandmarks() const;
    
    const LandmarkPtr& getLandmarkPtr(int i) const;
    const Pixel&       getPixel0(int i) const;
    const Pixel&       getPixel1(int i) const;
    float              getScaleChange(int i) const;

    const LandmarkPtrVec& getLandmarkPtrVec() const;
    const PixelVec&       getPixelVec0() const;
    const PixelVec&       getPixelVec1() const;
    const FloatVec&       getScaleChangeVec() const;

    LandmarkPtrVec& getLandmarkPtrVecRef();
    PixelVec&       getPixelVec0Ref();
    PixelVec&       getPixelVec1Ref();
    FloatVec&       getScaleChangeVecRef();

public:
    void setLandmarkPtrVec(const LandmarkPtrVec& lmvec);
    void setPixelVec0(const PixelVec& pts0);
    void setPixelVec1(const PixelVec& pts1);
    void setScaleChangeVec(const FloatVec& scale_change);


};

#endif