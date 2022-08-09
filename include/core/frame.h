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

    PoseSE3 dT10_;
    PoseSE3 dT01_;

    float steering_angle_;
    float scale_;

    cv::Mat image_;
    cv::Mat image_du_;
    cv::Mat image_dv_;

    LandmarkPtrVec related_landmarks_;

    PixelVec pts_seen_;

    double timestamp_;

    bool is_keyframe_;
    bool is_keyframe_in_window_;
    bool is_turning_frame_;
    
public: // static counter.
    inline static uint32_t frame_counter_ = 0;
    static std::shared_ptr<Camera> cam_;

public:
    Frame();

    // Set
    void setPose(const Eigen::Matrix4f& Twc);
    void setPoseDiff10(const Eigen::Matrix4f& dT10);
    void setSteeringAngle(float st_angle);
    void setScale(float scale);
    void makeThisKeyframe();
    void outOfKeyframeWindow();
    void makeThisTurningFrame();
    void setImageAndTimestamp(const cv::Mat& img, const double& timestamp); 
    void setRelatedLandmarks(const LandmarkPtrVec& landmarks);
    void setPtsSeen(const PixelVec& pts);

    // Get
    const uint32_t& getID() const;
    const PoseSE3& getPose() const;
    const PoseSE3& getPoseInv() const;
    const PoseSE3& getPoseDiff10() const;
    const PoseSE3& getPoseDiff01() const;
    const float& getSteeringAngle() const;
    const float& getScale() const;
    const cv::Mat& getImage() const ; 
    const cv::Mat& getImageDu() const ; 
    const cv::Mat& getImageDv() const ; 
    const LandmarkPtrVec& getRelatedLandmarkPtr() const;
    const PixelVec& getPtsSeen() const;
    const double& getTimestamp() const;
    bool isKeyframe() const;
    bool isKeyframeInWindow() const;
    bool isTurningFrame() const;
};

#endif