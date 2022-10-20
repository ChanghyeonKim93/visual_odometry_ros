#include "core/landmark_tracking.h"

LandmarkTracking::LandmarkTracking()
{
    pts0_.reserve(1000);
    pts1_.reserve(1000);
    lms_.reserve(1000);
    scale_change_.reserve(1000);
    n_pts_ = 0;
};

LandmarkTracking::LandmarkTracking(const LandmarkTracking& lmtrack, const MaskVec& mask)
{
    int n_pts_input = lmtrack.getNumLandmarks();

    std::vector<int> index_valid;
    index_valid.reserve(n_pts_input);

    int cnt_alive = 0;
    for(int i = 0; i < n_pts_input; ++i)
    {
        if( mask[i] )
        {
            index_valid.push_back(i);
            ++cnt_alive;
        }
        else
            lmtrack.getLandmarkPtr(i)->setDead();
    }

    // set
    n_pts_ = cnt_alive;

    pts0_.resize(n_pts_);
    pts1_.resize(n_pts_);
    lms_.resize(n_pts_);
    scale_change_.resize(n_pts_);

    for(int i = 0; i < n_pts_; ++i)
    {
        const int& idx = index_valid[i];
        pts0_[i]         = lmtrack.getPixel0(idx);
        pts1_[i]         = lmtrack.getPixel1(idx);
        lms_[i]          = lmtrack.getLandmarkPtr(idx);
        scale_change_[i] = lmtrack.getScaleChange(idx);
    }
};

void LandmarkTracking::initializeFromFramePtr(const FramePtr& f)
{
    const PixelVec&       pts0_frame = f->getPtsSeen();
    const LandmarkPtrVec&  lms_frame = f->getRelatedLandmarkPtr();

    int n_pts_frame = pts0_frame.size();
    
    pts0_.reserve(n_pts_frame);
    lms_.reserve(n_pts_frame);
    pts1_.reserve(n_pts_frame); 
    scale_change_.reserve(n_pts_frame);

    // Get only valid points
    for(int i = 0; i < n_pts_frame; ++i)
    {
        const LandmarkPtr& lm = lms_frame[i];

        if( lm->isAlive() )
        {
            pts0_.emplace_back(pts0_frame[i]);
            lms_.push_back(lm);
        }
    }

    n_pts_ = pts0_.size();

    pts1_.resize(n_pts_);
    scale_change_.resize(n_pts_);
};



const LandmarkPtr& LandmarkTracking::getLandmarkPtr(int i) const
{
    return lms_.at(i);
};
const Pixel&       LandmarkTracking::getPixel0(int i) const
{
    return pts0_.at(i);
};
const Pixel&       LandmarkTracking::getPixel1(int i) const
{
    return pts1_.at(i);
};
float       LandmarkTracking::getScaleChange(int i) const
{
    return scale_change_.at(i);
};

const LandmarkPtrVec& LandmarkTracking::getLandmarkPtrVec() const
{
    return lms_;
};
const PixelVec&       LandmarkTracking::getPixelVec0() const
{
    return pts0_;
};
const PixelVec&       LandmarkTracking::getPixelVec1() const
{
    return pts1_;
};
const FloatVec&       LandmarkTracking::getScaleChangeVec() const
{
    return scale_change_;
};

int LandmarkTracking::getNumLandmarks() const
{
    return n_pts_;
};


LandmarkPtrVec& LandmarkTracking::getLandmarkPtrVecRef()
{
    return lms_;
};

PixelVec&       LandmarkTracking::getPixelVec0Ref()
{
    return pts0_;
};

PixelVec&       LandmarkTracking::getPixelVec1Ref()
{
    return pts1_;
};

FloatVec&       LandmarkTracking::getScaleChangeVecRef()
{
    return scale_change_;
};


void LandmarkTracking::setLandmarkPtrVec(const LandmarkPtrVec& lmvec)
{
    lms_.resize(lmvec.size());
    std::copy(lmvec.begin(), lmvec.end(), lms_.begin());
    
};

void LandmarkTracking::setPixelVec0(const PixelVec& pts0)
{
    pts0_.resize(pts0.size());
    std::copy(pts0.begin(), pts0.end(), pts0_.begin());
};

void LandmarkTracking::setPixelVec1(const PixelVec& pts1)
{
    pts1_.resize(pts1.size());
    std::copy(pts1.begin(), pts1.end(), pts1_.begin());
};

void LandmarkTracking::setScaleChangeVec(const FloatVec& scale_change)
{
    scale_change_.resize(scale_change.size());
    std::copy(scale_change.begin(), scale_change.end(), scale_change_.begin());
};

