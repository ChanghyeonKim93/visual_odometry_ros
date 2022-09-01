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
    balms.reserve(3000);
};
LandmarkParameters::~LandmarkParameters(){

};

void LandmarkParameters::addLandmark(const LandmarkPtr& lm){
    BALandmarkPtr balm = std::make_shared<BALandmark>();

    // 최적화 대상인 keyframe을 추려낸다.
    // balm->addRelatedPixelAndFrame()
    const FramePtrVec& kfs_related = lm->getRelatedKeyframePtr();
    const PixelVec& pts_related    = lm->getObservationsOnKeyframes();
    for(int jj = 0; jj < kfs_related.size(); ++jj){
        const FramePtr& kf = kfs_related.at(jj);
        const Pixel&    pt = pts_related.at(jj);
        if(kf->isKeyframeInWindow()) // window keyframes
            balm->addRelatedPixelAndFrame(pt, kf);
    } // set related points & keyframes.

    // When # of related keyframe > 1, insert.
    if(balm->getRelatedKeyframes().size() > 1){
        balm->set3DPoint(lm->get3DPoint()); // set 3D point
        balm->setLandmarkPtr(lm); // set LandmarkPtr

        // Add this ba landmark.
        this->balms.push_back(balm);
        this->balm2idx.insert({balm, M_opt});
        this->lm2idx.insert({lm, M_opt});
        this->idx2balm.push_back(balm);

        ++M_opt;
        ++M;

        std::cout << "# of BA landmark to be optimized: "<< M_opt << std::endl;
    }
};

const BALandmarkPtr& LandmarkParameters::getBALandamrkPtrFromIndex(int i){ // get i-th BALandmark
    return balms.at(i);
};
const BALandmarkPtr& LandmarkParameters::getBALandamrkPtrFromOptIndex(int i_opt){ // get i-th optimizable BALandmark
    return idx2balm.at(i_opt);
};
int LandmarkParameters::getOptimizeIndex(const LandmarkPtr& lm){ // get optimization index from LandmarkPtr
    if(lm2idx.find(lm) != lm2idx.end() ) // this is a opt. keyframe.
        throw std::runtime_error("lm2idx.find(lm) != lm2idx.end()");
    return lm2idx[lm];
};
int LandmarkParameters::getOptimizeIndex(const BALandmarkPtr& balm){ // get Optimization index from BALandmarkPtr
    if(balm2idx.find(balm) != balm2idx.end() ) // this is a opt. keyframe.
        throw std::runtime_error("balm2idx.find(balm) != balm2idx.end()");
    return balm2idx[balm];
};


/*
 KeyframeParameters
  =============================================================================

*/
KeyframeParameters::KeyframeParameters() : N(0), N_opt(0)  {
    bakfs.reserve(300);
};

KeyframeParameters::~KeyframeParameters(){

};

void KeyframeParameters::addKeyframe(const FramePtr& kf, bool is_optimizable)
{

};
