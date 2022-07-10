#include "frame.h"

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

void Frame::setImage(const cv::Mat& img) { 
    img.copyTo(image_); 
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

// Check if a landmark is in the frustum of the frame.
// and fill variables of the landmark to be used by the tracking.
// 'viewing_cos_limit' is typically 0.5; (+-30 degrees ? )
bool Frame::isInFrustum(LandmarkPtr lm, float viewing_cos_limit){
    
    lm->setTrackInView(false);

    // 3D in absolute coordinates
    Eigen::Vector3f Xw = lm->get3DPoint();

    Eigen::Matrix3f Rcw = Tcw_.block<3,3>(0,0);
    Eigen::Vector3f tcw = Tcw_.block<3,1>(0,3);

    // 3D in camera coordinates
    const Eigen::Vector3f twc = Twc_.block<3,1>(0,3);
    const Eigen::Vector3f Xc  = Rcw*Xw + tcw;
    const float& xc = Xc(0);
    const float& yc = Xc(1);
    const float& zc = Xc(2);

    // Check positive depth
    if(zc < 0.0f) return false;

    // Project in image and check it is not outside
    const float invz = 1.0f/zc;
    const float u = fx*xc*invz + cx;
    const float v = fy*yc*invz + cy;

    if(u<mnMinX || u>mnMaxX) return false;
    if(v<mnMinY || v>mnMaxY) return false;

    // Check distance is in the scale invariance region of the MapPoint
    const float maxDistance = lm->GetMaxDistanceInvariance();
    const float minDistance = lm->GetMinDistanceInvariance();
    const Eigen::Vector3f X0 = Xw - twc; // vector from the camera center to the landmark
    const float dist = X0.norm(); // distance from the camera

    if(dist < minDistance || dist > maxDistance) return false;

   // Check viewing angle
    Eigen::Vector3f Pn = lm->GetNormal(); // normal vector of the landmark when firstly observed.
    // Pos.copyTo(mWorldPos);
    // cv::Mat Ow = pFrame->GetCameraCenter();
    // mNormalVector = mWorldPos - Ow;
    // mNormalVector = mNormalVector/cv::norm(mNormalVector);


    const float view_cos = X0.dot(Pn) / dist;

    if(view_cos < viewing_cos_limit) return false;

    // Predict scale in the image
    const int predict_scale_lvl = lm->PredictScale(dist, this);

    // Data used by the tracking
    lm->setTrackInView(true);
    lm->setTrackProjUV(u,v); // projection onto the currently focusing frame.
    lm->setTrackScaleLevel(predict_scale_lvl);
    lm->setTrackViewCos(view_cos);

    return true;
};
