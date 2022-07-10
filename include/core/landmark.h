#ifndef _LANDMARK_H_
#define _LANDMARK_H_

#include <iostream>
#include <vector>
#include <memory>
#include <Eigen/Dense>

#include <opencv2/core/core.hpp>

#include "core/frame.h"

/*
- Image frame :
    address of visual landmarks observed in the frame.
    a 6-DoF motion w.r.t. the global frame {W}.
*/
class Frame;
class Landmark;

typedef std::shared_ptr<Frame> FramePtr;
typedef std::shared_ptr<Landmark> LandmarkPtr;

class Landmark{
private:
    uint32_t id_;
    Eigen::Vector3f Xw_;

    std::vector<cv::KeyPoint> observations_;
    std::vector<FramePtr> related_frames_;

// Used for tracking
private:
    bool track_in_view_;
    float track_proj_u_;
    float track_proj_v_;
    uint32_t track_scale_level_;
    float track_view_cos_;

    float max_possible_distance_;
    float min_possible_distance_;

    Eigen::Vector3f normal_vector_;

public: // static counter
    inline static uint32_t landmark_counter_ = 0;

public:
    Landmark();

    void set3DPoint(const Eigen::Vector3f& Xw);
    void addObservationAndRelatedFrame(const cv::KeyPoint& p, const FramePtr& frame);

    void setTrackInView(bool value);
    void setTrackProjUV(float u, float v);
    void setTrackScaleLevel(uint32_t lvl);
    void setTrackViewCos(float vcos);
    
    uint32_t getID() const;
    Eigen::Vector3f get3DPoint() const;
    std::vector<cv::KeyPoint> getObservations() const;
    std::vector<FramePtr> getRelatedFramePtr() const;
};
#endif