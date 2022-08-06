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
    Point    x_front_;

    float invd_;
    float cov_invd_;

    PixelVec observations_;
    FramePtrVec related_frames_;

    bool is_alive_;
    bool is_triangulated_;

    uint32_t age_;

    float min_parallax_;
    float max_parallax_;
    float avg_parallax_;
    float last_parallax_;

    float min_optflow_;
    float max_optflow_;
    float avg_optflow_;
    float last_optflow_;

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
    void setInverseDepth(float invd_curr);
    void setCovarianceInverseDepth(float cov_invd_curr);
    void updateInverseDepth(float invd_curr, float cov_invd_curr);
    void setDead();
    
    // void setTrackInView(bool value);
    // void setTrackProjUV(float u, float v);
    // void setTrackScaleLevel(uint32_t lvl);
    // void setTrackViewCos(float vcos);
    
    uint32_t           getID() const;
    uint32_t           getAge() const;
    float              getInverseDepth() const;
    float              getCovarianceInverseDepth() const;
    const Point&       get3DPoint() const;
    const PixelVec&    getObservations() const;
    const FramePtrVec& getRelatedFramePtr() const;
    
    const bool&        isAlive() const;
    const bool&        isTriangulated() const;

    float              getMinParallax() const;  
    float              getMaxParallax() const;  
    float              getAvgParallax() const;  
    float              getLastParallax() const;  

    float              getMinOptFlow() const;
    float              getMaxOptFlow() const;
    float              getAvgOptFlow() const;
    float              getLastOptFlow() const;
};
#endif