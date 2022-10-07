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
#include "util/geometry_library.h"

/// @brief Image frame class
class Frame 
{
private:
    std::shared_ptr<Camera> cam_; // camera class pointer

private:
    uint32_t id_; // unique ID of this frame.

    double timestamp_; // timestamp of this frame in second.

    // Poses of the frame
    PoseSE3 Twc_; // SE(3) w.r.t. the global frame {W}.
    PoseSE3 Tcw_; // Inverse of Twc_.

    PoseSE3 dT10_;
    PoseSE3 dT01_;

    PoseSE3 dTkc_; // SE(3) from the related keyframe to the current.

    float steering_angle_;
    float scale_;

    // Images of the frame
    cv::Mat image_; // image (uint8)
    cv::Mat image_float_; // image (float32)
    cv::Mat image_du_; // derivative image of image along pixel u-axis.
    cv::Mat image_dv_; // derivative image of image along pixel v-axis.

    // Image pyramid for the pixel coordinate refinement
    ImagePyramid image_float_pyramid_;
    ImagePyramid image_du_pyramid_;
    ImagePyramid image_dv_pyramid_;

    // Feature related storages
    LandmarkPtrVec related_landmarks_;
    PixelVec pts_seen_;


    bool is_keyframe_;
    bool is_keyframe_in_window_;
    bool is_turning_frame_;

// For stereo (right) frame
private:
    bool is_right_image_; // stereo image.
    FramePtr frame_left_; // Frame pointer to 'left frame' of this right frame. In this project, we consider the 'left image' is the reference image of the stereo frames. All poses of the right frames are represented by their 'left frames'.
    
public: // static counter.
    inline static uint32_t frame_counter_ = 0; // Unique frame counter. (static)
    inline static int max_pyr_lvl_ = 5;

public:
    Frame(const std::shared_ptr<Camera>& cam, bool is_right_image, const FramePtr& frame_left);

    // Set
    void setPose(const PoseSE3& Twc);
    void setPoseDiff10(const PoseSE3& dT10);
    void setImageAndTimestamp(const cv::Mat& img, const double& timestamp); 
    void setSteeringAngle(float st_angle);
    void setScale(float scale);

    void setPtsSeenAndRelatedLandmarks(const PixelVec& pts, const LandmarkPtrVec& landmarks);
    void setPoseDiffFromLastKeyframe(const PoseSE3& dTkc);

    void makeThisKeyframe();
    void makeThisTurningFrame();
    void outOfKeyframeWindow();

    // Get
    const uint32_t& getID() const;
    const PoseSE3& getPose() const;
    const PoseSE3& getPoseInv() const;
    const PoseSE3& getPoseDiff10() const;
    const PoseSE3& getPoseDiff01() const;
    const float& getSteeringAngle() const;
    const float& getScale() const;
    const cv::Mat& getImage() const ; 
    const cv::Mat& getImageFloat() const ; 
    const cv::Mat& getImageDu() const ; 
    const cv::Mat& getImageDv() const ; 
    const LandmarkPtrVec& getRelatedLandmarkPtr() const;
    const PixelVec& getPtsSeen() const;
    const double& getTimestamp() const;
    bool isKeyframe() const;
    bool isKeyframeInWindow() const;
    bool isTurningFrame() const;

    const PoseSE3& getPoseDiffFromLastKeyframe() const;
    
};

#endif