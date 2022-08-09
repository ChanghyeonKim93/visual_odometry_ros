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
    std::list<FramePtr>  kfs_list_;
    std::vector<FramePtr> all_keyframes_;
    int n_max_keyframes_;

private:
    float THRES_OVERLAP_FEATURE_RATIO_;
    float THRES_ROTATION_;
    float THRES_TRANSLATION_;
    
public:
    Keyframes();

    void addNewKeyframe(const FramePtr& frame);
    bool checkUpdateRule(const FramePtr& frame_curr);

public:
    void setMaxKeyframes(int max_kf);

public:
    const std::list<FramePtr>& getList() const;
    int getCurrentNumOfKeyframes() const;
    int getMaxNumOfKeyframes() const;
};

#endif