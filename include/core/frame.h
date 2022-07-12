#ifndef _FRAME_H_
#define _FRAME_H_

#include <iostream>
#include <vector>
#include <memory>
#include <Eigen/Dense>

#include <opencv2/core/core.hpp>

#include "core/type_defines.h"

#include "core/landmark.h"
#include "core/camera.h"

/*
- Landmark
    2d pixel point hisotry over image
    Address of fraems where the landmark was seen.
    3D coordinate of the landmark represented in the global frame. It can be obtained by scale propagation and recovery modules.
*/

class Frame{
private:
    uint32_t id_;

    PoseSE3 Twc_;
    PoseSE3 Tcw_;
    cv::Mat image_;

    LandmarkPtrVec related_landmarks_;

    PixelVec pts_seen_;

    double timestamp_;

    bool is_keyframe_;
    
public: // static counter.
    inline static uint32_t frame_counter_ = 0;
    static std::shared_ptr<Camera> cam_;

public:
    Frame();

    // Set
    void setPose(const Eigen::Matrix4f& Twc);
    void setImageAndTimestamp(const cv::Mat& img, const double& timestamp); 
    void setRelatedLandmarks(const LandmarkPtrVec& landmarks);
    void setPtsSeen(const PixelVec& pts);
    void makeThisKeyframe();

    // Get
    const uint32_t& getID() const;
    const PoseSE3& getPose() const;
    const cv::Mat& getImage() const ; 
    const LandmarkPtrVec& getRelatedLandmarkPtr() const;
    const PixelVec& getPtsSeen() const;
    const double& getTimestamp() const;
    bool isKeyframe() const;
};

#endif