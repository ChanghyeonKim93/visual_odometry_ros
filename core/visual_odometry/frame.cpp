#include "core/visual_odometry/frame.h"

Frame::Frame(const std::shared_ptr<Camera>& cam, bool is_right_image, const FramePtr& frame_left)
: is_keyframe_(false), is_keyframe_in_window_(false), is_turning_frame_(false), is_poseonlyBA_success_(true)
{
    cam_ = cam;

    Twc_ = PoseSE3::Identity();
    Tcw_ = PoseSE3::Identity();

    dT01_ = PoseSE3::Identity();
    dT10_ = PoseSE3::Identity();

    steering_angle_ = 0.0f;
    scale_          = 0.0f;
    timestamp_      = 0.0;
    id_             = frame_counter_++;

// Stereo right image only.
    is_right_image_ = is_right_image;
    frame_left_     = frame_left;
};

Frame::Frame(const std::shared_ptr<Camera>& cam, const cv::Mat& img, const double& timestamp,
         bool is_right_image, const FramePtr& frame_left)
: is_keyframe_(false), is_keyframe_in_window_(false), is_turning_frame_(false), is_poseonlyBA_success_(true)
{
    cam_ = cam;

    Twc_ = PoseSE3::Identity();
    Tcw_ = PoseSE3::Identity();

    dT01_ = PoseSE3::Identity();
    dT10_ = PoseSE3::Identity();

    steering_angle_ = 0.0f;
    scale_          = 0.0f;
    timestamp_      = 0.0;
    id_             = frame_counter_++;

    this->setImageAndTimestamp(img, timestamp);

// Stereo right image only.
    is_right_image_ = is_right_image;
    frame_left_     = frame_left;

};


void Frame::setPose(const PoseSE3& Twc) { 
    Twc_ = Twc; 
    Tcw_ = geometry::inverseSE3_f(Twc_);
};

void Frame::setPoseDiff10(const Eigen::Matrix4f& dT10){
    dT10_ = dT10;
    dT01_ = geometry::inverseSE3_f(dT10);
};

void Frame::setImageAndTimestamp(const cv::Mat& img, const double& timestamp) { 
    img.copyTo(image_); // CV_8UC1
    img.convertTo(image_float_,CV_32FC1);// CV_32FC1
    timestamp_ = timestamp;

    // image_du_ = cv::Mat::zeros(image_.size(), CV_32FC1);
    // image_dv_ = cv::Mat::zeros(image_.size(), CV_32FC1);
    
    // Calculate diff images
    int kerner_size = 3;
    cv::Sobel(image_, image_du_, CV_32FC1, 1, 0, kerner_size, 1.0, 0.0, cv::BORDER_DEFAULT);
    cv::Sobel(image_, image_dv_, CV_32FC1, 0, 1, kerner_size, 1.0, 0.0, cv::BORDER_DEFAULT);
};

void Frame::setPtsSeenAndRelatedLandmarks(const PixelVec& pts, const LandmarkPtrVec& landmarks){
    if(pts.size() != landmarks.size())
        throw std::runtime_error("In 'Frame::setPtsSeenAndRelatedLandmarks()', pts.size() != landmarks.size()");

    // pts_seen
    pts_seen_.resize(pts.size());
    std::copy(pts.begin(), pts.end(), pts_seen_.begin());

    // related landmarks
    related_landmarks_.resize(landmarks.size());
    std::copy(landmarks.begin(), landmarks.end(), related_landmarks_.begin());
};

void Frame::makeThisKeyframe(){
    is_keyframe_           = true;
    is_keyframe_in_window_ = true;
};

void Frame::outOfKeyframeWindow(){
    is_keyframe_in_window_ = false;
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

const cv::Mat& Frame::getImage() const {
    return image_; 
}; 

const cv::Mat& Frame::getImageFloat() const {
    return image_float_; 
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
bool Frame::isKeyframeInWindow() const{
    return is_keyframe_in_window_;
};

bool Frame::isRightImage() const {
    return is_right_image_;
};

const FramePtr& Frame::getLeftFramePtr() const
{
    if(!is_right_image_) 
        throw std::runtime_error("In Frame::getLeftFramePtr() : This frame is not a right frame of a stereo camera!");
    
    return frame_left_;
};




// Related to turning frame
void Frame::makeThisTurningFrame(const FramePtr& frame_previous_turning)
{
    frame_previous_turning_ = frame_previous_turning;
    is_turning_frame_ = true;
};

void Frame::setSteeringAngle(float steering_angle){
    steering_angle_ = steering_angle;
};

void Frame::setScaleRaw(float scale_raw){
    scale_raw_ = scale_raw;
};

void Frame::setScale(float scale){
    scale_ = scale;
};

const FramePtr& Frame::getPreviousTurningFrame() const
{
    return frame_previous_turning_;
};

const float& Frame::getSteeringAngle() const{
    return steering_angle_;
};

const float& Frame::getScaleRaw() const {
    return scale_raw_;
};

const float& Frame::getScale() const {
    return scale_;
};

bool Frame::isTurningFrame() const {
    return is_turning_frame_;
};

void Frame::cancelThisTurningFrame()
{
    is_turning_frame_       = false;
    frame_previous_turning_ = nullptr;
    steering_angle_         = 0.0;
    scale_raw_              = 0.0;
    scale_                  = 0.0;
};


bool Frame::isPoseOnlySuccess() const
{
    return is_poseonlyBA_success_;
};
void Frame::setPoseOnlyFailed()
{
    is_poseonlyBA_success_ = false;
};






/*
====================================================================================
====================================================================================
=============================  StereoFrame  =============================
====================================================================================
====================================================================================
*/

// stereo frame

StereoFrame::StereoFrame(const FramePtr& f_l, const FramePtr& f_r)
{
    left_ = f_l;
    right_ = f_r;
};

StereoFrame::StereoFrame(const cv::Mat& img_left, const cv::Mat& img_right, CameraConstPtr& cam_left, CameraConstPtr& cam_right, double timestamp)
{
	left_  = std::make_shared<Frame>(cam_left, img_left, timestamp);
	right_ = std::make_shared<Frame>(cam_right, img_right, timestamp, true, left_);
};


FrameConstPtr& StereoFrame::getLeft() const
{
    return left_;
};

FrameConstPtr& StereoFrame::getRight() const
{
    return right_;
};

void StereoFrame::setStereoPoseByLeft(const PoseSE3& Twc_left, const PoseSE3& T_lr)
{
    left_->setPose(Twc_left);
    right_->setPose(Twc_left * T_lr);
};

void StereoFrame::setStereoPtsSeenAndRelatedLandmarks(const PixelVec& pts_l1, const PixelVec& pts_r1, const LandmarkPtrVec& lms)
{
    left_->setPtsSeenAndRelatedLandmarks(pts_l1, lms);
    right_->setPtsSeenAndRelatedLandmarks(pts_r1, lms);
};  