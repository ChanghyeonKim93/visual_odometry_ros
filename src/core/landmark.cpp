#include "core/landmark.h"

std::shared_ptr<Camera> Landmark::cam_ = nullptr;

Landmark::Landmark()
: Xw_(0,0,0), id_(landmark_counter_++), max_possible_distance_(0),min_possible_distance_(0), age_(1), is_alive_(true), is_triangulated_(false), max_parallax_(0.0)
{
    
};  
Landmark::Landmark(const Pixel& p, const FramePtr& frame)
: Xw_(0,0,0), id_(landmark_counter_++), max_possible_distance_(0),min_possible_distance_(0), age_(1), is_alive_(true), is_triangulated_(false), max_parallax_(0.0)
{
    addObservationAndRelatedFrame(p, frame);
};  

Landmark::~Landmark(){
    std::cout << "Landmark destructor called.\n";  
};

void Landmark::set3DPoint(const Point& Xw) { Xw_ = Xw;  is_triangulated_ = true; };
void Landmark::addObservationAndRelatedFrame(const Pixel& p, const FramePtr& frame) {
    observations_.push_back(p);
    related_frames_.push_back(frame);
    if(observations_.size() != related_frames_.size()){
        throw std::runtime_error("observeation.size() != related_frames_.size(). please check.");
    }
    ++age_;

    // Calculate parallax w.r.t. the oldest pixel
    const Pixel& p0 = observations_.front();
    const Pixel& p1 = observations_.back();

    Point x0, x1;
    x0 << (p0.x-cam_->cx())*cam_->fxinv(), (p0.y-cam_->cy())*cam_->fyinv(), 1.0f; 
    x1 << (p1.x-cam_->cx())*cam_->fxinv(), (p1.y-cam_->cy())*cam_->fyinv(), 1.0f; 

    float costheta = x0.dot(x1)/(x0.norm()*x1.norm());
    if(costheta >  1) costheta =  0.999f;
    if(costheta < -1) costheta = -0.999f;
    
    float parallax_curr = acos(costheta);
    if(max_parallax_ < parallax_curr) max_parallax_ = parallax_curr;
};    

// void Landmark::setTrackInView(Mask value){
//     track_in_view_ = value;
// };
// void Landmark::setTrackProjUV(float u, float v){
//     track_proj_u_ = u; track_proj_v_ = v;
// };
// void Landmark::setTrackScaleLevel(uint32_t lvl){
//     track_scale_level_ = lvl;
// };
// void Landmark::setTrackViewCos(float vcos){
//     track_view_cos_ = vcos;
// };

void               Landmark::setDead() { is_alive_ = false; };

uint32_t           Landmark::getID() const   { return id_; };
uint32_t           Landmark::getAge() const  { return age_; };
float              Landmark::getMaxParallax() const  { return max_parallax_; };
const Point&       Landmark::get3DPoint() const { return Xw_; };
const PixelVec&    Landmark::getObservations() const { return observations_; };
const FramePtrVec& Landmark::getRelatedFramePtr() const { return related_frames_; };
const bool&        Landmark::getAlive() const { return is_alive_; };
const bool&        Landmark::getTriangulated() const { return is_triangulated_; };