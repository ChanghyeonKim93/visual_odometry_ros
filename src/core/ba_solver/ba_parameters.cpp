#include "core/ba_solver/ba_parameters.h"

/*
 BALandmark
  =============================================================================

*/
BALandmark::BALandmark() : Xw(0,0,0), landmark_ptr(nullptr) {
    pts_related.reserve(50);
    kfs_related.reserve(50);
};

BALandmark::~BALandmark(){};
void BALandmark::setLandmarkPtr(const LandmarkPtr& lmptr) { 
    landmark_ptr = lmptr;
};
void BALandmark::addRelatedPixelAndFrame(const Pixel& pt, const FramePtr& kf) {
    pts_related.emplace_back(pt);
    kfs_related.emplace_back(kf);
};
void BALandmark::set3DPoint(const Point& X) {
    Xw = X; 
};

int BALandmark::getSize()                               { return pts_related.size(); };
const PixelVec&    BALandmark::getRelatedPixels()       { return pts_related; };
const FramePtrVec& BALandmark::getRelatedKeyframes()    { return kfs_related; };
const Point&       BALandmark::get3DPoint()             { return Xw; };
const LandmarkPtr& BALandmark::getOriginalLandmarkPtr() { return landmark_ptr; };



/*
 BAKeyframe
  =============================================================================

*/
 BAKeyframe::BAKeyframe() : is_optimizable(false), index_opt(-1), frame_ptr(nullptr) {
    lms_related.reserve(500);
    pose_jw = PoseSE3::Identity();
};
BAKeyframe::~BAKeyframe(){};
void BAKeyframe::setKeyframePtr(const FramePtr& kfptr){
    frame_ptr = kfptr;
};
void BAKeyframe::addRelatedLandmark(const LandmarkPtr& lm){
    lms_related.emplace_back(lm);
};
void BAKeyframe::setPose(const PoseSE3& Tjw){
    pose_jw = Tjw;
};
void BAKeyframe::setOptimizableWithIndex(int idx){
    index_opt      = idx;
    is_optimizable = true;
};
bool                  BAKeyframe::isOptimizable()       { return is_optimizable; };
int                   BAKeyframe::getIndexOpt()         { return index_opt; };
const LandmarkPtrVec& BAKeyframe::getRelatedLandmarks() { return lms_related; };
const PoseSE3&        BAKeyframe::getPose()             { return pose_jw; };
const FramePtr&       BAKeyframe::getOriginalFramePtr() { return frame_ptr; };



/*
 LandmarkParameters
  =============================================================================

*/
LandmarkParameters::LandmarkParameters(): M(0), M_opt(0){
    lms.reserve(3000);
};
LandmarkParameters::~LandmarkParameters(){

};

void LandmarkParameters::addLandmark(const LandmarkPtr& lm){
    BALandmarkPtr balm_ptr = std::make_shared<BALandmark>();

    // balm_ptr->addRelatedPixelAndFrame()
};


const BALandmarkPtr& LandmarkParameters::getLandamrkParameter(int i){
    return lms[i];
};
int LandmarkParameters::getIndexOfLandmark(const BALandmarkPtr& lm){
    return lms2idx[lm];
};



/*
 KeyframeParameters
  =============================================================================

*/
KeyframeParameters::KeyframeParameters() : N(0), N_opt(0)  {
    kfs.reserve(300);
};

KeyframeParameters::~KeyframeParameters(){

};

void KeyframeParameters::addKeyframeParameter(const FramePtr& kf, bool is_optimizable){
    BAKeyframePtr bakf_ptr = std::make_shared<BAKeyframe>();
    bakf_ptr->setKeyframePtr(kf);
    bakf_ptr->setPose(kf->getPoseInv());

    // bakf_ptr->addRelatedLandmark;
    
    if(is_optimizable == true){

        bakf_ptr->setOptimizableWithIndex(N_opt++);

    }
    else { // non-optimizable keyframe

    }

    kfs.push_back(bakf_ptr);
    ++N;
};

const BAKeyframePtr& KeyframeParameters::getKeyframeParameter(int j){
    return kfs[j];
};
