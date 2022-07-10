#ifndef _FRAME_H_
#define _FRAME_H_

#include <iostream>
#include <vector>
#include <memory>
#include <Eigen/Dense>

#include <opencv2/core/core.hpp>

#include "core/landmark.h"

#define FRAME_GRID_COLS 64
#define FRAME_GRID_ROWS 48
/*
- Landmark
    2d pixel point hisotry over image
    Address of fraems where the landmark was seen.
    3D coordinate of the landmark represented in the global frame. It can be obtained by scale propagation and recovery modules.
*/
class Frame;
class Landmark;

typedef std::shared_ptr<Frame> FramePtr;
typedef std::shared_ptr<Landmark> LandmarkPtr;

class Frame{
private:
    uint32_t id_;

    Eigen::Matrix4f Twc_;
    Eigen::Matrix4f Tcw_;
    cv::Mat image_;

    std::vector<LandmarkPtr> related_landmarks_;

    double timestamp_;

private:
    std::vector<uint32_t> feature_grid_[FRAME_GRID_ROWS][FRAME_GRID_COLS];
    cv::Mat discriptor_;
    
public: // static counter.
    inline static uint32_t frame_counter_ = 0;

public:
    Frame();

    // Set
    void setPose(const Eigen::Matrix4f& Twc);
    void setImageAndTimestamp(const cv::Mat& img, const double& timestamp); 
    void setLandmarks(const std::vector<LandmarkPtr>& landmarks);
    
    // Get
    uint32_t getID() const;
    Eigen::Matrix4f getPose() const;
    cv::Mat getImage() const ; 
    std::vector<LandmarkPtr> getRelatedLandmarkPtr() const;
    double getTimestamp() const;
};

#endif