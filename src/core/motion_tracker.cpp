#include "core/motion_tracker.h"

MotionTracker::MotionTracker(){

};

MotionTracker::~MotionTracker(){
    
};

cv::Mat MotionTracker::trackCurrentImage(const cv::Mat& img, const double& timestamp){
    // get motion...
    this->img_current_ = img;
    if(img_current_.channels() != 1) {
        throw std::runtime_error("grabImageMonocular - Image is not grayscale image.");
    }

    if(track_state_ == TrackingState::NOT_INITIALIZED || track_state_ == TrackingState::NO_IMAGES_YET){
        // Initialize the current frame
    }
    else {
        
    }
    
    // track the input image 
    track();
    
    // return current_frame_.Tcw.clone();
};  

void MotionTracker::track(){
    
};

void MotionTracker::monocularInitialization(){

};