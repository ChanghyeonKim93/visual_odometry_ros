#include "core/frame.h"
std::shared_ptr<Camera> Frame::cam_ = nullptr;

Frame::Frame() : is_keyframe_(false) {
    Twc_ = PoseSE3::Identity();
    Tcw_ = PoseSE3::Identity();
    steering_angle_ = 0.0f;
    id_  = frame_counter_;
    timestamp_ = 0;
    ++frame_counter_;
};

void Frame::setPose(const PoseSE3& Twc) { 
    Twc_ = Twc; 
    Tcw_ = Twc_.inverse();
};
void Frame::setSteeringAngle(float st_angle){
    steering_angle_ = st_angle;
};

void Frame::setImageAndTimestamp(const cv::Mat& img, const double& timestamp) { 
    img.copyTo(image_); 
    timestamp_ = timestamp;
};

void Frame::setRelatedLandmarks(const LandmarkPtrVec& landmarks){
    related_landmarks_.resize(0);
    related_landmarks_.reserve(landmarks.size());
    for(auto lm : landmarks) related_landmarks_.push_back(lm);
};

void Frame::setPtsSeen(const PixelVec& pts){
    pts_seen_.resize(pts.size());
    std::copy(pts.begin(),pts.end(), pts_seen_.begin());
};

void Frame::makeThisKeyframe(){
    is_keyframe_ = true;
};

const uint32_t& Frame::getID() const { 
    return id_; 
};

const PoseSE3& Frame::getPose() const { 
    return Twc_; 
};
const float& Frame::getSteeringAngle() const{
    return steering_angle_;
};

const cv::Mat& Frame::getImage() const {
    return image_; 
}; 

const LandmarkPtrVec& Frame::getRelatedLandmarkPtr() const { 
    return related_landmarks_; 
};

const PixelVec& Frame::getPtsSeen() const {
    return pts_seen_;  
};

const double& Frame::getTimestamp() const {
    return timestamp_;
};

bool Frame::isKeyframe() const{
    return is_keyframe_;
};