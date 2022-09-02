#include "core/landmark.h"

std::shared_ptr<Camera> Landmark::cam_ = nullptr;

Landmark::Landmark()
: id_(landmark_counter_++), age_(0), 
Xw_(0,0,0), x_front_(0,0,0), 
invd_(0.0f), cov_invd_(100000.0f), 
is_alive_(true), is_triangulated_(false), is_bundled_(false)
{
    // Reserve storages
    observations_.reserve(200);
    related_frames_.reserve(200);

    observations_on_keyframes_.reserve(50);
    related_keyframes_.reserve(50);

    // Initialize parallax and opt flow.
    min_parallax_ = 1000.0f;
    max_parallax_ = 0.0f;
    avg_parallax_ = 0.0f;
    last_parallax_ = 0.0f;

    min_optflow_ = 1000.0f;
    max_optflow_ = 0.0f;
    avg_optflow_ = 0.0f;
    last_optflow_ = 0.0f;
};

Landmark::Landmark(const Pixel& p, const FramePtr& frame)
: id_(landmark_counter_++), age_(0), 
Xw_(0,0,0), x_front_(0,0,0), 
invd_(0.0f), cov_invd_(100000.0f), 
is_alive_(true), is_triangulated_(false), is_bundled_(false)
{
    // Reserve storages
    observations_.reserve(200);
    related_frames_.reserve(200);

    observations_on_keyframes_.reserve(50);
    related_keyframes_.reserve(50);
    
    // normalized coordinate
    x_front_(0)= ( p.x-cam_->cx() )*cam_->fxinv();
    x_front_(1)= ( p.y-cam_->cy() )*cam_->fyinv();
    x_front_(2)= 1.0f;

    // Initialize parallax and opt flow.
    min_parallax_  = 1000.0f;
    max_parallax_  = 0.0f;
    avg_parallax_  = 0.0f;
    last_parallax_ = 0.0f;

    min_optflow_   = 1000.0f;
    max_optflow_   = 0.0f;
    avg_optflow_   = 0.0f;
    last_optflow_  = 0.0f;

    // Add observation
    this->addObservationAndRelatedFrame(p, frame);
};  

Landmark::~Landmark(){
    std::cout << "Landmark destructor called.\n";  
};

void Landmark::set3DPoint(const Point& Xw) { 
    Xw_ = Xw;  is_triangulated_ = true; 
};
void Landmark::setBundled() { 
    is_bundled_ = true; 
};
void Landmark::setInverseDepth(float invd_curr) { 
    invd_ = invd_curr; 
};
void Landmark::setCovarianceInverseDepth(float cov_invd_curr) { 
    cov_invd_ = cov_invd_curr; 
};
void Landmark::updateInverseDepth(float invd_curr, float cov_invd_curr)
{
    // new std
    float invd_new = (invd_*cov_invd_curr + invd_curr*cov_invd_)/(cov_invd_ + cov_invd_curr);
    invd_ = invd_new;

    float cov_new = cov_invd_curr*cov_invd_/(cov_invd_curr+cov_invd_);
    cov_invd_ = cov_new;


    Point Xf;
    Xf = x_front_ * (1.0f/invd_);

    const PoseSE3& Twf = related_frames_.front()->getPose();
    Xw_ = Twf.block<3,3>(0,0)*Xf + Twf.block<3,1>(0,3);
};

void Landmark::addObservationAndRelatedFrame(const Pixel& p, const FramePtr& frame) {
    // push observation.
    ++age_;
    observations_.push_back(p);
    related_frames_.push_back(frame);
    
    if(observations_.size() == 1) return;
    
    // Calculate parallax w.r.t. the oldest pixel
    // const Pixel& p0 = observations_[observations_.size()-2];
    const Pixel& p0 = observations_.front();
    const Pixel& p1 = observations_.back();

    // const PoseSE3& T01 = related_frames_[observations_.size()-2]->getPose().inverse()*related_frames_.back()->getPose();
    PoseSE3 T01 = related_frames_.front()->getPoseInv()*related_frames_.back()->getPose();

    Point x0, x1;
    x0 << (p0.x-cam_->cx())*cam_->fxinv(), (p0.y-cam_->cy())*cam_->fyinv(), 1.0f; 
    x1 << (p1.x-cam_->cx())*cam_->fxinv(), (p1.y-cam_->cy())*cam_->fyinv(), 1.0f; 
    x1 = T01.block<3,3>(0,0)*x1;

    float costheta = x0.dot(x1)/(x0.norm()*x1.norm());
    if(costheta >=  1) costheta =  0.999f;
    if(costheta <= -1) costheta = -0.999f;

    float parallax_curr = acos(costheta);
    last_parallax_ = parallax_curr;
    if(max_parallax_ <= parallax_curr) max_parallax_ = parallax_curr;
    if(min_parallax_ >= parallax_curr) min_parallax_ = parallax_curr;

    float invage = 1.0f/(float)age_;
    avg_parallax_ = avg_parallax_*(float)(age_-1.0f);
    avg_parallax_ += parallax_curr;
    avg_parallax_ *= invage;

    // Calculate optical flow 
    Pixel dp = p1-p0;
    float optflow_now = sqrt(dp.x*dp.x + dp.y*dp.y);
    last_optflow_ = optflow_now;
    if(optflow_now >= max_optflow_) max_optflow_ = optflow_now;
    if(optflow_now <= min_optflow_) min_optflow_ = optflow_now; 

    avg_optflow_ = avg_optflow_*(float)(age_-1.0f);
    avg_optflow_ += optflow_now;
    avg_optflow_ *= invage;

};    

void Landmark::addObservationAndRelatedKeyframe(const Pixel& p, const FramePtr& kf){
    observations_on_keyframes_.push_back(p);
    related_keyframes_.push_back(kf);
};

void               Landmark::setDead()                  { is_alive_ = false; };

uint32_t           Landmark::getID() const              { return id_; };
uint32_t           Landmark::getAge() const             { return age_; };
float              Landmark::getInverseDepth() const    { return invd_; };
float              Landmark::getCovarianceInverseDepth() const { return cov_invd_; };
const Point&       Landmark::get3DPoint() const         { return Xw_; };
const PixelVec&    Landmark::getObservations() const    { return observations_; };
const FramePtrVec& Landmark::getRelatedFramePtr() const { return related_frames_; };
const PixelVec&    Landmark::getObservationsOnKeyframes() const { return observations_on_keyframes_; };
const FramePtrVec& Landmark::getRelatedKeyframePtr() const      { return related_keyframes_; };

const bool&        Landmark::isAlive() const           { return is_alive_; };
const bool&        Landmark::isTriangulated() const    { return is_triangulated_; };
const bool&        Landmark::isBundled() const         { return is_bundled_; };

float              Landmark::getMinParallax() const     { return min_parallax_; };
float              Landmark::getMaxParallax() const     { return max_parallax_; };
float              Landmark::getAvgParallax() const     { return avg_parallax_; };
float              Landmark::getLastParallax() const    { return last_parallax_; };

float              Landmark::getMinOptFlow() const      { return min_optflow_; };
float              Landmark::getMaxOptFlow() const      { return max_optflow_; };
float              Landmark::getAvgOptFlow() const      { return avg_optflow_; };
float              Landmark::getLastOptFlow() const     { return last_optflow_; };