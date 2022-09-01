#ifndef _BA_PARAMETERS_H_
#define _BA_PARAMETERS_H_

#include <iostream>
#include <vector>
#include <map>
#include <set>

#include <Eigen/Dense>

#include "core/camera.h"
#include "core/landmark.h"
#include "core/frame.h"
#include "core/type_defines.h"

#include "util/geometry_library.h"
#include "util/timer.h"
/* 
    <Optimization variables>
        
    BundleAdjustmentVariables ba_vars;
    ba_vars.landmark.lms[i].pts_related

    landmark.lms_to_index (std::map, LandmarkPtr -> index)
    landmark.index_to_lms (std::map, index -> LandmarkPtr)
    landmark.lms[i]
        - pts_related (std::vector)
        - kfs_related (std::vector)
        - Xw (3D point)
        - landmark_ptr


    keyframe.kfs_to_index (FramePtr -> index)
    keyframe.index_to_kfs (index ->FramePtr)
    keyframe.kfs[j]
        - lms_related (std::vector)
        - pose_jw (4x4 matrix)
        - is_optimizable (bool)
        - index_optimize (if optimizable, )
        - frame_ptr
*/

class BALandmark;
class BAKeyframe;

class LandmarkParameters;
class KeyframeParameters;

class BundleAdjustmentVariables;

typedef std::shared_ptr<BALandmark> BALandmarkPtr;
typedef std::shared_ptr<BAKeyframe> BAKeyframePtr;


/*
 BALandmark
  =============================================================================

*/
class BALandmark{
private:
    PixelVec    pts_related;  // pixel observations
    FramePtrVec kfs_related;  // keyframe observations
    Point       Xw;           // 3D point represented in world frame.

    LandmarkPtr landmark_ptr; // the original landmark ptr.

public:
    BALandmark();
    ~BALandmark();

// Set methods
public:
    void setLandmarkPtr(const LandmarkPtr& lmptr);
    void addRelatedPixelAndFrame(const Pixel& pt, const FramePtr& kf);
    void set3DPoint(const Point& X);

// Get methods
public:
    int getSize();
    const PixelVec&    getRelatedPixels();
    const FramePtrVec& getRelatedKeyframes();
    const Point&       get3DPoint();

    const LandmarkPtr& getOriginalLandmarkPtr();
};

/*
 BAKeyframe
  =============================================================================

*/
class BAKeyframe{
private:
    LandmarkPtrVec lms_related;
    PoseSE3        pose_jw;
    bool           is_optimizable;
    int            index_opt;

    FramePtr       frame_ptr;
    
public:
    BAKeyframe();
    ~BAKeyframe();

// Set methods
public:
    void setKeyframePtr(const FramePtr& kfptr);
    void addRelatedLandmark(const LandmarkPtr& lm);
    void setPose(const PoseSE3& Tjw);
    void setOptimizableWithIndex(int idx);

// Get methods
public:
    bool                  isOptimizable();
    int                   getIndexOpt();
    const LandmarkPtrVec& getRelatedLandmarks();
    const PoseSE3&        getPose();

    const FramePtr&       getOriginalFramePtr();
};


/*
 LandmarkParameters
  =============================================================================

*/
class LandmarkParameters{
private:
    int M; // the number of all landmarks.
    int M_opt; // the number of optimizable landmarks. (now, M == M_opt)

private:
    std::vector<BALandmarkPtr> balms;      // all BA landmarks (M)
    
    std::map<BALandmarkPtr,int> balm2idx;  // BALandmarkPtr to index (M_opt)
    std::map<LandmarkPtr,int>   lm2idx;    // LandmarkPtr to index (M_opt)
    std::vector<BALandmarkPtr>  idx2balm;  // index to BALandmarkPtr (M_opt)
    
public:
    LandmarkParameters();
    ~LandmarkParameters();

// Set methods
public:
    void addLandmark(const LandmarkPtr& lm); // Add a landmark (generate BALandmark)

// Get methods
public:
    const BALandmarkPtr& getBALandamrkPtrFromIndex(int i); // get i-th BALandmark
    const BALandmarkPtr& getBALandamrkPtrFromOptIndex(int i_opt); // get i-th optimizable BALandmark
    int getOptimizeIndex(const LandmarkPtr& lm); // get optimization index from LandmarkPtr
    int getOptimizeIndex(const BALandmarkPtr& balm); // get Optimization index from BALandmarkPtr
};

/*
 KeyframeParameters
  =============================================================================

*/
class KeyframeParameters{
private:
    int N; // the number of total frames
    int N_opt; // the number of optimizable frames

private:
    std::vector<BAKeyframePtr> bakfs;     // all BAKeyframePtr (N)

    std::map<BAKeyframePtr,int> bakf2idx; // BAKeyframePtr to index (N_opt)
    std::map<FramePtr,int>      kf2idx;   // FramePtr to index (N_opt)
    std::vector<BAKeyframePtr>  idx2bakf; // index to BAKeyframePtr (N_opt)

public:
    KeyframeParameters();
    ~KeyframeParameters();

public:
    void addKeyframe(const FramePtr& kf, bool is_optimizable);

public:
    const BAKeyframePtr& getKeyframeParameter(int j);
    bool isOptimizableKeyframe(const FramePtr& kf);
    bool isOptimizableBAKeyframe(const BAKeyframePtr& bakf);
};

/*
 BundleAdjustmentVariables
  =============================================================================

*/
class BundleAdjustmentVariables{
private:
    LandmarkParameters landmarks;
    KeyframeParameters keyframes;

public:
    BundleAdjustmentVariables(){
            
    };
    ~BundleAdjustmentVariables(){

    };
};

    
#endif