#ifndef _LANDMARK_H_
#define _LANDMARK_H_

#include <iostream>
#include <vector>
#include <memory>
#include <Eigen/Dense>

#include <opencv2/core/core.hpp>

#include "core/type_defines.h"

#include "core/frame.h"
#include "core/camera.h"

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

    uint32_t age_;

    float min_parallax_;
    float max_parallax_;
    float avg_parallax_;

    float min_optflow_;
    float max_optflow_;
    float avg_optflow_;


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
    static std::shared_ptr<Camera> cam_;

public:
    Landmark();
    Landmark(const Pixel& p, const FramePtr& frame);
    ~Landmark();

    void addObservationAndRelatedFrame(const Pixel& p, const FramePtr& frame);
    
    void set3DPoint(const Point& Xw);
    void setDead();
    
    // void setTrackInView(bool value);
    // void setTrackProjUV(float u, float v);
    // void setTrackScaleLevel(uint32_t lvl);
    // void setTrackViewCos(float vcos);
    
    uint32_t           getID() const;
    uint32_t           getAge() const;
    const Point&       get3DPoint() const;
    const PixelVec&    getObservations() const;
    const FramePtrVec& getRelatedFramePtr() const;
    const bool&        getAlive() const;
    const bool&        getTriangulated() const;

    float              getMinParallax() const;  
    float              getMaxParallax() const;  
    float              getAvgParallax() const;  

    float              getMinOptFlow() const;
    float              getMaxOptFlow() const;
    float              getAvgOptFlow() const;
};
#endif