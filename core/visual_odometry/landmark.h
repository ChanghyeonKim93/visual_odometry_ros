#ifndef _LANDMARK_H_
#define _LANDMARK_H_

#include <iostream>
#include <vector>
#include <memory>

#include "eigen3/Eigen/Dense"

#include "opencv4/opencv2/core.hpp"

#include "core/defines/define_type.h"

#include "core/visual_odometry/frame.h"
#include "core/visual_odometry/camera.h"

class Landmark;
class LandmarkTracking;
class StereoLandmarkTracking;

// 2d pixel point hisotry over image
// Address of fraems where the landmark was seen.
// 3D coordinate of the landmark represented in the global frame. It can be obtained by scale propagation and recovery modules.

/// @brief Landmark class
class Landmark
{
private:
    uint32_t id_; // feature unique id
    Point    Xw_; // 3D point represented in the global frame.

    // scale refinement는 keyframe에서만 할까?
    std::vector<float> I0_patt_; // 최초로 관측 된 위치에서의 patch (image patch)
    std::vector<float> du0_patt_;// 최초로 관측 된 위치에서의 patch (derivative along u)
    std::vector<float> dv0_patt_;// 최초로 관측 된 위치에서의 patch (derivative along v)
    MaskVec mask_patt_; // valid mask
    
    PixelVec observations_; // 2D pixel observation history of this landmark
    std::vector<float> view_sizes_; // (approx.) 2D patch size compared to the firstly observed image. (s1 = s2*d1/d2)
    FramePtrVec related_frames_; // frame history where this landmark was seen

// keyframes
private:
    PixelVec observations_on_keyframes_; // 2D pixel observation history of this landmark
    FramePtrVec related_keyframes_; // frame history where this landmark was seen

// status
private:
    bool is_alive_;   // alive flag. 추적이 끝났고, bundle 등 결과, 점의 퀄리티가 너무 안좋으면 dead 로 만들어서 앞으로 사용하지 않는다. (ex- SQP 이후 점 좌표가 너무 커진 경우 등)
    bool is_tracked_; // track flag. 추적이 끝났음을 알리는 것. dead와는 무관하다. (local bundle & SQP 등에서 사용되기 때문.)
    bool is_triangulated_; // triangulated flag. 3D point를 가지고 있는지 여부. 
    bool is_bundled_; // bundled flag. Bundle에 참여했었는지 여부.

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
    Landmark(); // cosntructor
    Landmark(const Pixel& p, const FramePtr& frame); // constructor with observation.
    ~Landmark();// destructor

// Set methods
public:
    void addObservationAndRelatedFrame(const Pixel& p, const FramePtr& frame);
    void addObservationAndRelatedKeyframe(const Pixel& p, const FramePtr& frame);

    void changeLastObservation(const Pixel& p);
    
    void set3DPoint(const Point& Xw);
    
    void setUntracked();
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

    // const std::vector<float>& getImagePatchVec() const;
    // const std::vector<float>& getDuPatchVec() const;
    // const std::vector<float>& getDvPatchVec() const;
    // const MaskVec&            getMaskPatchVec() const;

    const bool&        isAlive() const;
    const bool&        isTracked() const;
    const bool&        isTriangulated() const;
    const bool&        isBundled() const;

    float              getMinParallax() const;  
    float              getMaxParallax() const;  
    float              getAvgParallax() const;  
    float              getLastParallax() const;  
};

class LandmarkTracking
{
public:
    PixelVec pts0;
    PixelVec pts1;
    LandmarkPtrVec lms;
    std::vector<float> scale_change;

    int n_pts;

// Various constructors
public:
    LandmarkTracking();
    LandmarkTracking(const LandmarkTracking& lmtrack, const MaskVec& mask);
    LandmarkTracking(const PixelVec& pts0_in, const PixelVec& pts1_in, const LandmarkPtrVec& lms_in);
};



/// @brief A temporal structur for stereo feature tracking 
class StereoLandmarkTracking 
{
public:
    //  left camera prev. and current.
    PixelVec pts_l0;
    PixelVec pts_l1;

    // right camera prev. and current.
    PixelVec pts_r0;
    PixelVec pts_r1;

    // LandmarkPtr vector.
    LandmarkPtrVec lms;

    int n_pts;

public:
    /// @brief Constructor of StereoLandmarkTracking
    StereoLandmarkTracking();
    StereoLandmarkTracking(const StereoLandmarkTracking& slmtrack, const MaskVec& mask);
};

#endif