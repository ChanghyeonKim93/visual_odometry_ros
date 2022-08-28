#ifndef _KEYFRAMES_H_
#define _KEYFRAMES_H_

#include <iostream>
#include <list>
#include <vector>
#include <set>

#include <memory>
#include <Eigen/Dense>

#include <opencv2/core/core.hpp>

#include "core/type_defines.h"
#include "core/defines.h"

#include "core/landmark.h"
#include "core/camera.h"
#include "core/frame.h"

class Keyframes{
private:
    std::list<FramePtr>  kfs_list_; // lits of window keyframes
    std::vector<FramePtr> all_keyframes_; // vector of all keyframe history

private:
    float THRES_OVERLAP_FEATURE_RATIO_; // feature overlap ratio to update new keyframe
    float THRES_ROTATION_; // rotation magnitude from the last keyframe
    float THRES_TRANSLATION_; // translation magnitude from the last keyframe to update 
    int   N_MAX_KEYFRAMES_IN_WINDOW_; // maximum number of keyframes in window.
    
public:
    Keyframes(); // constructor of keyframes class

public:
    void setMaxKeyframes(int max_kf); // set maximum keyframes in window.

public:
    bool checkUpdateRule(const FramePtr& frame_curr); // check keyframe update rule.
    void addNewKeyframe(const FramePtr& frame); // delete the oldest keyframe and add new keyframe.

public:
    const std::list<FramePtr>& getList() const; // get list of window keyframes
    int getCurrentNumOfKeyframes() const; // get current number of window keyframes
    int getMaxNumOfKeyframes() const; // get maximum allowed number of window keyframes
};

#endif