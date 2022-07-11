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

void Frame::setRelatedLandmarks(const LandmarkPtrVec& landmarks){
    related_landmarks_.resize(0);
    related_landmarks_.reserve(landmarks.size());
    for(auto lm : landmarks) {
        related_landmarks_.push_back(lm);
    }
};

void Frame::setPtsSeen(const PixelVec& pts){
    pts_seen_.resize(0);
    pts_seen_.reserve(pts.size());
    for(auto p : pts) {
        pts_seen_.emplace_back(p);
    }
};

const uint32_t& Frame::getID() const { 
    return id_; 
};

const Eigen::Matrix4f& Frame::getPose() const { 
    return Twc_; 
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