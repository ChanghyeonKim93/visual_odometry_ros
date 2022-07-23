#include "core/feature_tracker.h"

FeatureTracker::FeatureTracker(){
    printf(" - FEATURE_TRACKER is constructed.\n");
};

FeatureTracker::~FeatureTracker(){
    printf(" - FEATURE_TRACKER is deleted.\n");
};

void FeatureTracker::track(const cv::Mat& img0, const cv::Mat& img1, const PixelVec& pts0, 
    PixelVec& pts_track, MaskVec& mask_valid) {
    int n_cols = img0.size().width;
    int n_rows = img0.size().height;

    int n_pts = pts0.size();
    mask_valid.resize(n_pts, true);

    // KLT tracking
    pts_track.resize(0);
    pts_track.reserve(n_pts);

    std::vector<uchar> status;
    std::vector<float> err;
    int maxLevel = 4;
    cv::calcOpticalFlowPyrLK(img0, img1, 
        pts0, pts_track, 
        status, err, cv::Size(25,25), maxLevel);
    
    for(int i = 0; i < n_pts; ++i){
        mask_valid[i] = (mask_valid[i] && status[i] > 0);
    }
    
    printf(" - FEATURE_TRACKER - 'track()'\n");
};

void FeatureTracker::trackBidirection(const cv::Mat& img0, const cv::Mat& img1, const PixelVec& pts0, uint32_t window_size, uint32_t max_pyr_lvl, float thres_err, float thres_bidirection,
                PixelVec& pts_track, MaskVec& mask_valid)
{

    float thres_bidirection2 = thres_bidirection*thres_bidirection;

    int n_cols = img0.size().width;
    int n_rows = img0.size().height;

    int n_pts = pts0.size();
    mask_valid.resize(n_pts, true);

    // KLT tracking
    pts_track.resize(0);
    pts_track.reserve(n_pts);

    int maxLevel = max_pyr_lvl;

    // forward tracking
    std::vector<uchar> status_forward;
    std::vector<float> err_forward;
    cv::calcOpticalFlowPyrLK(img0, img1, 
        pts0, pts_track, 
        status_forward, err_forward, cv::Size(window_size,window_size), maxLevel+1);
    
    // backward tracking
    PixelVec pts0_backward(n_pts);
    std::copy(pts0.begin(), pts0.end(), pts0_backward.begin());
    std::vector<uchar> status_backward;
    std::vector<float> err_backward;
    cv::calcOpticalFlowPyrLK(img1, img0, 
        pts_track, pts0_backward,
        status_backward, err_backward, cv::Size(window_size,window_size), maxLevel-2, {}, cv::OPTFLOW_USE_INITIAL_FLOW, {});

    // Check validity.
    for(int i = 0; i < n_pts; ++i){
        Pixel dp = pts0_backward[i]-pts0[i];
        float dist2 = dp.x*dp.x + dp.y*dp.y;

        // border validity
        mask_valid[i] = (mask_valid[i] && pts_track[i].x > 3 && pts_track[i].x < n_cols-3
                                       && pts_track[i].y > 3 && pts_track[i].y < n_rows-3);
        // other ...
        mask_valid[i] = (mask_valid[i] 
            && status_forward[i]
            && status_backward[i]
            && err_forward[i]  <= thres_err
            && err_backward[i] <= thres_err
            && dist2 <= thres_bidirection2
        );
    }
    
    printf(" - FEATURE_TRACKER - 'trackBidirection()'\n");
};


void FeatureTracker::trackBidirectionWithPrior(const cv::Mat& img0, const cv::Mat& img1, const PixelVec& pts0, uint32_t window_size, const PixelVec& pts1_prior, float thres_err, float thres_bidirection,
                PixelVec& pts_track, MaskVec& mask_valid)
{

    float thres_bidirection2 = thres_bidirection*thres_bidirection;

    int n_cols = img0.size().width;
    int n_rows = img0.size().height;

    int n_pts = pts0.size();
    mask_valid.resize(n_pts, true);

    // KLT tracking
    pts_track.resize(0);
    pts_track.reserve(n_pts);
    std::copy(pts1_prior.begin(), pts1_prior.end(), pts_track.begin());

    int maxLevel = 5;

    // forward tracking
    std::vector<uchar> status_forward;
    std::vector<float> err_forward;
    cv::calcOpticalFlowPyrLK(img0, img1, 
        pts0, pts_track, 
        status_forward, err_forward, cv::Size(15,15), maxLevel, {}, cv::OPTFLOW_USE_INITIAL_FLOW, {});

    // backward tracking
    PixelVec pts0_backward(n_pts);
    std::copy(pts0.begin(), pts0.end(), pts0_backward.begin());
    std::vector<uchar> status_backward;
    std::vector<float> err_backward;
    cv::calcOpticalFlowPyrLK(img1, img0, 
        pts_track, pts0_backward,
        status_backward, err_backward, cv::Size(15,15), maxLevel, {}, cv::OPTFLOW_USE_INITIAL_FLOW, {});

    // Check validity.
    for(int i = 0; i < n_pts; ++i){
        Pixel dp = pts0_backward[i]-pts0[i];
        float dist2 = dp.x*dp.x + dp.y*dp.y;

        // border validity
        mask_valid[i] = (mask_valid[i] && pts_track[i].x > 0 && pts_track[i].x < n_cols
                                       && pts_track[i].y > 0 && pts_track[i].y < n_rows);
        // other ...
        mask_valid[i] = (mask_valid[i] 
            && status_forward[i]
            && status_backward[i]
            && err_forward[i]  <= thres_err
            && err_backward[i] <= thres_err
            && dist2 <= thres_bidirection2
        );
    }
    
    printf(" - FEATURE_TRACKER - 'trackBidirection()'\n");
};

void FeatureTracker::trackWithPrior(const cv::Mat& img0, const cv::Mat& img1, const PixelVec& pts0, const PixelVec& pts1_prior,
                PixelVec& pts_track, MaskVec& mask_valid) 
{
    int n_cols = img0.size().width;
    int n_rows = img0.size().height;

    int n_pts = pts0.size();
    mask_valid.resize(n_pts, true);

    pts_track.resize(n_pts);
    std::copy(pts1_prior.begin(), pts1_prior.end(), pts_track.begin());

    // KLT tracking with prior.
    std::vector<uchar> status;
    std::vector<float> err;
    int maxLevel = 4;
    cv::calcOpticalFlowPyrLK(img0, img1,
        pts0, pts_track,
        status, err, cv::Size(25,25), maxLevel, {}, cv::OPTFLOW_USE_INITIAL_FLOW, {});
    
    for(int i = 0; i < n_pts; ++i){
        mask_valid[i] = (mask_valid[i] && status[i] > 0);
    }
    printf(" - FEATURE_TRACKER - 'trackWithPrior()'\n");
};

void FeatureTracker::calcPrior(const PixelVec& pts0, const PointVec& Xw, const PoseSE3& Tw1, const Eigen::Matrix3f& K,
    PixelVec& pts1_prior) 
{
    int n_pts = Xw.size();
    pts1_prior.resize((int)(pts0.size()));
    std::copy(pts0.begin(),pts0.end(), pts1_prior.begin());

    Eigen::Matrix4f T1w = Tw1.inverse();
    PointVec X1;
    X1.reserve(n_pts);
    for(auto Xw_ : Xw) X1.emplace_back((T1w.block(0,0,3,3)*Xw_ + T1w.block(0,3,3,1)));

    for(int i = 0; i < n_pts; ++i)
    {   
        Point X = X1[i];
        int mask_haveprior = X.norm() > 0;
        if(mask_haveprior){
            Pixel pt_tmp;
            pt_tmp.x = K(0,0)*X(0)/X(2) + K(0,2);
            pt_tmp.y = K(1,1)*X(1)/X(2) + K(1,2); // projection
            pts1_prior[i] = pt_tmp ;
        }
        // else pts1_prior[i] = pts0[i];
    }
};