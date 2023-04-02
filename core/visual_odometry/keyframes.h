#ifndef _KEYFRAMES_H_
#define _KEYFRAMES_H_

#include <iostream>
#include <list>
#include <vector>
#include <set>
#include <memory>

#include <eigen3/Eigen/Dense>
#include "opencv4/opencv2/core.hpp"

#include "core/visual_odometry/define_macro.h"
#include "core/visual_odometry/define_type.h"

#include "core/visual_odometry/landmark.h"
#include "core/visual_odometry/camera.h"
#include "core/visual_odometry/frame.h"

class Keyframes;
class StereoKeyframes;

/// @brief Keyframe class
class Keyframes
{
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
    void setThresTranslation(float val);
    void setThresRotation(float val);
    void setThresOverlapRatio(float val);

public:
    bool checkUpdateRule(const FramePtr& frame_curr); // check keyframe update rule.
    void addNewKeyframe(const FramePtr& frame); // delete the oldest keyframe and add new keyframe.

public:
    const std::list<FramePtr>& getList() const; // get list of window keyframes
    int getCurrentNumOfKeyframes() const; // get current number of window keyframes
    int getMaxNumOfKeyframes() const; // get maximum allowed number of window keyframes
};




/// @brief Stereo Keyframe class
class StereoKeyframes
{
private:
    std::list<StereoFramePtr>   stereo_kfs_list_; // lits of window keyframes
    std::vector<StereoFramePtr> all_stereo_keyframes_; // vector of all keyframe history

private:
    float THRES_OVERLAP_FEATURE_RATIO_; // feature overlap ratio to update new keyframe
    float THRES_ROTATION_; // rotation magnitude from the last keyframe
    float THRES_TRANSLATION_; // translation magnitude from the last keyframe to update 
    int   N_MAX_KEYFRAMES_IN_WINDOW_; // maximum number of keyframes in window.
    
public:
    StereoKeyframes(); // constructor of keyframes class

public:
    void setMaxStereoKeyframes(int max_kf); // set maximum keyframes in window.
    void setThresTranslation(float val);
    void setThresRotation(float val); 
    void setThresOverlapRatio(float val); 

public:
    void addNewStereoKeyframe(const StereoFramePtr& stframe); // delete the oldest keyframe and add new keyframe.
    bool checkUpdateRule(const StereoFramePtr& stframe_curr); // check keyframe update rule.

public:
    const std::list<StereoFramePtr>& getList() const; // get list of window keyframes
    int getCurrentNumOfStereoKeyframes() const; // get current number of window keyframes
    int getMaxNumOfStereoKeyframes() const; // get maximum allowed number of window keyframes
};
#endif