#include "core/frame.h"

Frame::Frame(const std::shared_ptr<Camera>& cam, bool is_right_image = false, const FramePtr& frame_left = nullptr)
: is_keyframe_(false), is_keyframe_in_window_(false), is_turning_frame_(false) 
{
    cam_ = cam;

    Twc_ = PoseSE3::Identity();
    Tcw_ = PoseSE3::Identity();
    steering_angle_ = 0.0f;
    scale_          = 0.0f;
    timestamp_      = 0.0;
    id_             = frame_counter_++;

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
        throw std::runtime_error("pts.size() != landmarks.size()");

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
    is_turning_frame_ = false;
    frame_previous_turning_ = nullptr;
    steering_angle_ = 0.0;
    scale_raw_ = 0.0;
    scale_ = 0.0;
};