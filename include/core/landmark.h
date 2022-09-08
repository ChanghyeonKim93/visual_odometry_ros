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
class Landmark{
private:
    uint32_t id_; // feature unique id
    Point    Xw_; // 3D point represented in the global frame.
    Point    x_front_; // normalized 3D point represented in the first seen image.

    // scale refinement는 keyframe에서만 할까?
    std::vector<float> I0_patt_; // 최초로 관측 된 위치에서의 patch (image patch)
    std::vector<float> du0_patt_;// 최초로 관측 된 위치에서의 patch (derivative along u)
    std::vector<float> dv0_patt_;// 최초로 관측 된 위치에서의 patch (derivative along v)
    MaskVec mask_patt_; // valid mask
    
    float invd_; // inverse depth of the 3D point represented in the first seen image.
    float cov_invd_; // covariance of the inverse depth 

    PixelVec observations_; // 2D pixel observation history of this landmark
    std::vector<float> view_sizes_; // (approx.) 2D patch size compared to the firstly observed image. (s1 = s2*d1/d2)
    FramePtrVec related_frames_; // frame history where this landmark was seen

// keyframes
private:
    PixelVec observations_on_keyframes_; // 2D pixel observation history of this landmark
    FramePtrVec related_keyframes_; // frame history where this landmark was seen

// status
private:
    bool is_alive_; // alive flag
    bool is_triangulated_; // triangulated flag
    bool is_bundled_; // bundled flag

    uint32_t age_; // tracking age

    float min_parallax_; // the smallest parallax 
    float max_parallax_; // the largest parallax
    float avg_parallax_; // average parallax
    float last_parallax_; // the last parallax
 
    float min_optflow_;
    float max_optflow_;
    float avg_optflow_;
    float last_optflow_;

    // Eigen::Vector3f normal_vector_;

public: // static counter
    inline static uint32_t landmark_counter_ = 0; // unique id counter.
    static std::shared_ptr<Camera> cam_; // camera object.
    static PixelVec patt_;

    static void setPatch(int half_win_sz) {
        int win_sz = 2*half_win_sz + 1;
        int n_elem = win_sz*win_sz;

        patt_.resize(n_elem);
        int ind = 0;
        for(int v = 0; v < win_sz; ++v) {
            for(int u = !(v & 0x01); u < win_sz; u += 2) {
                patt_[ind].x = (float)(u - half_win_sz);
                patt_[ind].y = (float)(v - half_win_sz);
                ++ind;
            }
        }
        n_elem = ind;
        patt_.resize(n_elem);
        std::cout << "in setPatch, n_elem: " << n_elem << std::endl; 
    };


public:
    Landmark(); // cosntructor
    Landmark(const Pixel& p, const FramePtr& frame); // constructor with observation.
    ~Landmark();// destructor

// Set methods
public:
    void addObservationAndRelatedFrame(const Pixel& p, const FramePtr& frame);
    void addObservationAndRelatedKeyframe(const Pixel& p, const FramePtr& frame);

    void changeLastObservation(const Pixel& p);
    
    void set3DPoint(const Point& Xw);

    void setBundled();
    void setInverseDepth(float invd_curr);
    void setDead();
    
    void updateInverseDepth(float invd_curr, float cov_invd_curr);
    void setCovarianceInverseDepth(float cov_invd_curr);
    
    // void setTrackInView(bool value);
    // void setTrackProjUV(float u, float v);
    // void setTrackScaleLevel(uint32_t lvl);
    // void setTrackViewCos(float vcos);

// Get methods
public:
    uint32_t           getID() const;
    uint32_t           getAge() const;
    float              getInverseDepth() const;
    float              getCovarianceInverseDepth() const;
    const Point&       get3DPoint() const;
    const PixelVec&    getObservations() const;
    const PixelVec&    getObservationsOnKeyframes() const;
    const FramePtrVec& getRelatedFramePtr() const;
    const FramePtrVec& getRelatedKeyframePtr() const;

    const std::vector<float>& getImagePatchVec() const;
    const std::vector<float>& getDuPatchVec() const;
    const std::vector<float>& getDvPatchVec() const;
    const MaskVec&            getMaskPatchVec() const;

    const bool&        isAlive() const;
    const bool&        isTriangulated() const;
    const bool&        isBundled() const;

    float              getMinParallax() const;  
    float              getMaxParallax() const;  
    float              getAvgParallax() const;  
    float              getLastParallax() const;  

    float              getMinOptFlow() const;
    float              getMaxOptFlow() const;
    float              getAvgOptFlow() const;
    float              getLastOptFlow() const;
};

struct LandmarkTracking{
    PixelVec pts0;
    PixelVec pts1;
    LandmarkPtrVec lms;
    std::vector<float> scale_change;

    LandmarkTracking(){
        pts0.reserve(1000);
        pts1.reserve(1000);
        lms.reserve(1000);
        scale_change.reserve(1000);
    };
};

#endif