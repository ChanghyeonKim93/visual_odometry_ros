#include "core/frame.h"
std::shared_ptr<Camera> Frame::cam_ = nullptr;

Frame::Frame() : is_keyframe_(false), is_turning_frame_(false) {
    Twc_ = PoseSE3::Identity();
    Tcw_ = PoseSE3::Identity();
    steering_angle_ = 0.0f;
    scale_ = 0.0f;
    id_  = frame_counter_;
    timestamp_ = 0;
    ++frame_counter_;
};

void Frame::setPose(const PoseSE3& Twc) { 
    Twc_ = Twc; 
    Tcw_ = Twc_.inverse();
};

void Frame::setPoseDiff10(const Eigen::Matrix4f& dT10){
    dT10_ = dT10;
    dT01_ = dT10.inverse();
};

void Frame::setSteeringAngle(float st_angle){
    steering_angle_ = st_angle;
};

void Frame::setScale(float scale){
    scale_ = scale;
};

void Frame::setImageAndTimestamp(const cv::Mat& img, const double& timestamp) { 
    img.copyTo(image_); 
    timestamp_ = timestamp;

    // image_du_ = cv::Mat::zeros(image_.size(), CV_32FC1);
    // image_dv_ = cv::Mat::zeros(image_.size(), CV_32FC1);
    
    // Calculate diff images
    int kerner_size = 3;
    cv::Sobel(image_, image_du_, CV_32FC1, 1, 0, kerner_size, 1.0, 0.0, cv::BORDER_DEFAULT);
    cv::Sobel(image_, image_dv_, CV_32FC1, 0, 1, kerner_size, 1.0, 0.0, cv::BORDER_DEFAULT);
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
void Frame::makeThisTurningFrame(){
    is_turning_frame_ = true;
};
const uint32_t& Frame::getID() const { 
    return id_; 
};

const PoseSE3& Frame::getPose() const { 
    return Twc_; 
};
const PoseSE3& Frame::getPoseInv() const{
    return Tcw_;
};

const PoseSE3& Frame::getPoseDiff10() const{
    return dT10_;
};
const PoseSE3& Frame::getPoseDiff01() const{
    return dT01_;
};

const float& Frame::getSteeringAngle() const{
    return steering_angle_;
};
const float& Frame::getScale() const {
    return scale_;
}

const cv::Mat& Frame::getImage() const {
    return image_; 
}; 

const cv::Mat& Frame::getImageDu() const {
    return image_du_;
};

const cv::Mat& Frame::getImageDv() const {
    return image_dv_;
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
bool Frame::isTurningFrame() const {
    return is_turning_frame_;
};