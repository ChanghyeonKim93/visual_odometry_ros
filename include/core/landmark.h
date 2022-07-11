#ifndef _LANDMARK_H_
#define _LANDMARK_H_

#include <iostream>
#include <vector>
#include <memory>
#include <Eigen/Dense>

#include <opencv2/core/core.hpp>

#include "core/type_defines.h"

#include "core/frame.h"

/*
- Image frame :
    address of visual landmarks observed in the frame.
    a 6-DoF motion w.r.t. the global frame {W}.
*/
class Landmark;

class Landmark{
private:
    uint32_t id_;
    Point    Xw_;

    PixelVec observations_;
    FramePtrVec related_frames_;

    bool is_alive_;
    bool is_triangulated_;

// Used for tracking
private:
    // bool track_in_view_;
    // float track_proj_u_;
    // float track_proj_v_;
    // uint32_t track_scale_level_;
    // float track_view_cos_;

    float max_possible_distance_;
    float min_possible_distance_;

    // Eigen::Vector3f normal_vector_;

public: // static counter
    inline static uint32_t landmark_counter_ = 0;

public:
    Landmark();
    Landmark(const Pixel& p, const FramePtr& frame);
    ~Landmark();

    void set3DPoint(const Point& Xw);
    void addObservationAndRelatedFrame(const Pixel& p, const FramePtr& frame);

    // void setTrackInView(bool value);
    // void setTrackProjUV(float u, float v);
    // void setTrackScaleLevel(uint32_t lvl);
    // void setTrackViewCos(float vcos);
    void setAlive(bool value);
    
    const uint32_t& getID() const;
    const Point& get3DPoint() const;
    const PixelVec& getObservations() const;
    const FramePtrVec& getRelatedFramePtr() const;
    const bool& getAlive() const;
    const bool& getTriangulated() const;
};
#endif