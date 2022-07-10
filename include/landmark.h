#ifndef _LANDMARK_H_
#define _LANDMARK_H_

#include <iostream>
#include <vector>
#include <memory>
#include <Eigen/Dense>

#include <opencv2/core/core.hpp>

#include "frame.h"

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
    Landmark()
    : Xw_(0,0,0), id_(landmark_counter_++), max_possible_distance_(0),min_possible_distance_(0)
    {
        
    };  

    void set3DPoint(const Eigen::Vector3f& Xw) { Xw_ = Xw; };
    void addObservationAndRelatedFrame(const cv::KeyPoint& p, const FramePtr& frame) {
        observations_.push_back(p);
        related_frames_.push_back(frame);
        if(observations_.size() != related_frames_.size()){
            throw std::runtime_error("observeation.size() != related_frames_.size(). please check.");
        }
    };    

    void setTrackInView(bool value){
        track_in_view_ = value;
    };
    void setTrackProjUV(float u, float v){
        track_proj_u_ = u; track_proj_v_ = v;
    };
    void setTrackScaleLevel(uint32_t lvl){
        track_scale_level_ = lvl;
    };
    void setTrackViewCos(float vcos){
        track_view_cos_ = vcos;
    };

    uint32_t predictScaleLevel(const float& current_dist, const FramePtr& frame){
        float ratio;
        ratio = max_possible_distance_/current_dist;

        int n_scale_lvl = ceil(log(ratio)/frame->getLogScaleFactor());
        if(n_scale_lvl < 0) 
            n_scale_lvl = 0;
        else if(n_scale_lvl >= frame->mnScaleLevels)
            n_scale_lvl = frame->mnScaleLevels-1;

        return n_scale_lvl;
    };

    
    uint32_t getID() const { return id_; };
    Eigen::Vector3f get3DPoint() const { return Xw_; };
    std::vector<cv::KeyPoint> getObservations() const { return observations_; };
    std::vector<FramePtr> getRelatedFramePtr() const { return related_frames_; };
};
#endif