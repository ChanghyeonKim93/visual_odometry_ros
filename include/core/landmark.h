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

// 2d pixel point hisotry over image
// Address of fraems where the landmark was seen.
// 3D coordinate of the landmark represented in the global frame. It can be obtained by scale propagation and recovery modules.

/// @brief Landmark class
class Landmark
{
private:
    uint32_t id_; // feature unique id
    Point    Xw_; // 3D point represented in the global frame.
    Point    x_front_; // normalized 3D point represented in the first seen image.

    // scale refinement는 keyframe에서만 할까?
    std::vector<float> I0_patt_; // 최초로 관측 된 위치에서의 patch (image patch)
    std::vector<float> du0_patt_;// 최초로 관측 된 위치에서의 patch (derivative along u)
    std::vector<float> dv0_patt_;// 최초로 관측 된 위치에서의 patch (derivative along v)
    MaskVec mask_patt_; // valid mask
    
    PixelVec observations_; // 2D pixel observation history of this landmark
    std::vector<float> view_sizes_; // (approx.) 2D patch size compared to the firstly observed image. (s1 = s2*d1/d2)
    FramePtrVec related_frames_; // frame history where this landmark was seen
  
    std::shared_ptr<Camera> cam_; // camera object.

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
 
    // Eigen::Vector3f normal_vector_;

public: // static counter
    inline static uint32_t landmark_counter_ = 0; // unique id counter.
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
    Landmark(const std::shared_ptr<Camera>& cam); // cosntructor
    Landmark(const Pixel& p, const FramePtr& frame, const std::shared_ptr<Camera>& cam); // constructor with observation.
    ~Landmark();// destructor

// Set methods
public:
    void addObservationAndRelatedFrame(const Pixel& p, const FramePtr& frame);
    void addObservationAndRelatedKeyframe(const Pixel& p, const FramePtr& frame);

    void changeLastObservation(const Pixel& p);
    
    void set3DPoint(const Point& Xw);

    void setBundled();
    void setDead();
    
// Get methods
public:
    uint32_t           getID() const;
    uint32_t           getAge() const;
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
};

/// @brief A temporal structur for monocular feature tracking 
struct LandmarkTracking
{
    PixelVec pts0;
    PixelVec pts1;
    LandmarkPtrVec lms;
    std::vector<float> scale_change;

    LandmarkTracking()
    {
        pts0.reserve(1000);
        pts1.reserve(1000);
        lms.reserve(1000);
        scale_change.reserve(1000);
    };
};

/// @brief A temporal structur for stereo feature tracking 
struct StereoLandmarkTracking 
{
    // lower camera prev. and current.
    PixelVec pts_l0;
    PixelVec pts_l1;

    // upper camera prev. and current.
    PixelVec pts_u0;
    PixelVec pts_u1;

    // LandmarkPtr vector.
    LandmarkPtrVec lms;

    /// @brief Constructor of StereoLandmarkTracking
    StereoLandmarkTracking()
    {
        pts_l0.reserve(1000); pts_l1.reserve(1000);
        pts_u0.reserve(1000); pts_u1.reserve(1000);
    
        lms.reserve(1000);
    };
};

#endif