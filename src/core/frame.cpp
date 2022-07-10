#include "core/frame.h"

Frame::Frame() {
    Twc_ = Eigen::Matrix4f::Identity();
    Tcw_ = Eigen::Matrix4f::Identity();
    id_  = frame_counter_;
    ++frame_counter_;
};

void Frame::setPose(const Eigen::Matrix4f& Twc) { 
    Twc_ = Twc; 
    Tcw_ = Twc_.inverse();
};

void Frame::setImageAndTimestamp(const cv::Mat& img, const double& timestamp) { 
    img.copyTo(image_); 
    timestamp_ = timestamp;
};

void Frame::setLandmarks(const std::vector<LandmarkPtr>& landmarks){
    related_landmarks_.resize(0);
    related_landmarks_.reserve(landmarks.size());
    for(auto lm : landmarks) related_landmarks_.push_back(lm);
};

uint32_t Frame::getID() const { 
    return id_; 
};

Eigen::Matrix4f Frame::getPose() const { 
    return Twc_; 
};

cv::Mat Frame::getImage() const {
    return image_; 
}; 

std::vector<LandmarkPtr> Frame::getRelatedLandmarkPtr() const { 
    return related_landmarks_; 
};
double Frame::getTimestamp() const {
    return timestamp_;
};