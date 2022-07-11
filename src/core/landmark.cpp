#include "core/landmark.h"

Landmark::Landmark()
: Xw_(0,0,0), id_(landmark_counter_++), max_possible_distance_(0),min_possible_distance_(0), is_alive_(true), is_triangulated_(false)
{
    
};  
Landmark::Landmark(const Pixel& p, const FramePtr& frame)
: Xw_(0,0,0), id_(landmark_counter_++), max_possible_distance_(0),min_possible_distance_(0), is_alive_(true), is_triangulated_(false)
{
    addObservationAndRelatedFrame(p, frame);
};  

void Landmark::set3DPoint(const Point& Xw) { Xw_ = Xw;  is_triangulated_ = true; };
void Landmark::addObservationAndRelatedFrame(const Pixel& p, const FramePtr& frame) {
    observations_.push_back(p);
    related_frames_.push_back(frame);
    if(observations_.size() != related_frames_.size()){
        throw std::runtime_error("observeation.size() != related_frames_.size(). please check.");
    }
};    

void Landmark::setTrackInView(Mask value){
    track_in_view_ = value;
};
void Landmark::setTrackProjUV(float u, float v){
    track_proj_u_ = u; track_proj_v_ = v;
};
void Landmark::setTrackScaleLevel(uint32_t lvl){
    track_scale_level_ = lvl;
};
void Landmark::setTrackViewCos(float vcos){
    track_view_cos_ = vcos;
};
void Landmark::setAlive(bool value){
    is_alive_ = value;
};

const uint32_t& Landmark::getID() const { 
    return id_; 
};
const Point& Landmark::get3DPoint() const { return Xw_; };
const PixelVec& Landmark::getObservations() const { return observations_; };
const FramePtrVec& Landmark::getRelatedFramePtr() const { return related_frames_; };
const bool& Landmark::getAlive() const { return is_alive_; };
const bool& Landmark::getTriangulated() const { return is_triangulated_; };