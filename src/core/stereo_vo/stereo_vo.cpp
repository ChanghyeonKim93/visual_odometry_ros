#include "core/stereo_vo/stereo_vo.h"

StereoVO::StereoVO(std::string mode, std::string directory_intrinsic)
: cam_left_(nullptr), cam_right_(nullptr), stframe_prev_(nullptr)
{
	std::cout << "Stereo VO starts\n";
		
	// Initialize camera
	cam_left_  = std::make_shared<Camera>();
	cam_right_ = std::make_shared<Camera>();

	if(mode == "dataset")
	{
		throw std::runtime_error("dataset mode is not supported.");
	}
	else if(mode == "rosbag")
	{
		std::cout << "StereoVO - 'rosbag' mode.\n";
		
		this->loadStereoCameraIntrinsicAndUserParameters(directory_intrinsic);
		// wait callback ...
	}
	else 
		throw std::runtime_error("StereoVO - unknown mode.");

	// Initialize feature extractor (ORB-based)
	extractor_ = std::make_shared<FeatureExtractor>();
	int n_bins_u     = params_.feature_extractor.n_bins_u;
	int n_bins_v     = params_.feature_extractor.n_bins_v;
	float THRES_FAST = params_.feature_extractor.thres_fastscore;
	float radius     = params_.feature_extractor.radius;
	extractor_->initParams(cam_left_->cols(), cam_left_->rows(), n_bins_u, n_bins_v, THRES_FAST, radius);

	// Initialize feature tracker (KLT-based)
	tracker_ = std::make_shared<FeatureTracker>();

	// Initialize motion estimator
	motion_estimator_ = std::make_shared<MotionEstimator>();
	motion_estimator_->setThres1p(params_.motion_estimator.thres_1p_error);
	motion_estimator_->setThres5p(params_.motion_estimator.thres_5p_error);

	// Initialize stereo keyframes class
	stkeyframes_ = std::make_shared<StereoKeyframes>();
};

void StereoVO::loadStereoCameraIntrinsicAndUserParameters(const std::string& dir)
{
    cv::FileStorage fs(dir, cv::FileStorage::READ);
	if (!fs.isOpened()) throw std::runtime_error("stereo intrinsic file cannot be found!\n");

// Left camera
	int rows, cols;
	rows = fs["Camera.left.height"];	cols = fs["Camera.left.width"];

	float fx, fy, cx, cy;
	fx = fs["Camera.left.fx"];	fy = fs["Camera.left.fy"];
	cx = fs["Camera.left.cx"];	cy = fs["Camera.left.cy"];

	float k1,k2,k3,p1,p2;
	k1 = fs["Camera.left.k1"];	k2 = fs["Camera.left.k2"];	k3 = fs["Camera.left.k3"];
	p1 = fs["Camera.left.p1"];	p2 = fs["Camera.left.p2"];

	cv::Mat cvK_tmp;
	cvK_tmp = cv::Mat(3,3,CV_32FC1);
	cvK_tmp.at<float>(0,0) = fx;	cvK_tmp.at<float>(0,1) = 0.0f;	cvK_tmp.at<float>(0,2) = cx;
	cvK_tmp.at<float>(1,0) = 0.0f;	cvK_tmp.at<float>(1,1) = fy;	cvK_tmp.at<float>(1,2) = cy;
	cvK_tmp.at<float>(2,0) = 0.0f;	cvK_tmp.at<float>(2,1) = 0.0f;	cvK_tmp.at<float>(2,2) = 1.0f;
	
	cv::Mat cvD_tmp;
	cvD_tmp = cv::Mat(1,5,CV_32FC1);
	cvD_tmp.at<float>(0,0) = k1;
	cvD_tmp.at<float>(0,1) = k2;
	cvD_tmp.at<float>(0,2) = p1;
	cvD_tmp.at<float>(0,3) = p2;
	cvD_tmp.at<float>(0,4) = k3;

	if(cam_left_ == nullptr) 
        throw std::runtime_error("cam_left_ is not allocated.");

	cam_left_->initParams(cols, rows, cvK_tmp, cvD_tmp);

	std::cout << "fx_l: " << cam_left_->fx() <<", "
			  << "fy_l: " << cam_left_->fy() <<", "
			  << "cx_l: " << cam_left_->cx() <<", "
			  << "cy_l: " << cam_left_->cy() <<", "
			  << "cols_l: " << cam_left_->cols() <<", "
			  << "rows_l: " << cam_left_->rows() <<"\n";
// Right camera
	rows = fs["Camera.right.height"];	cols = fs["Camera.right.width"];

	fx = fs["Camera.right.fx"];	fy = fs["Camera.right.fy"];
	cx = fs["Camera.right.cx"];	cy = fs["Camera.right.cy"];

	k1 = fs["Camera.right.k1"];	k2 = fs["Camera.right.k2"];	k3 = fs["Camera.right.k3"];
	p1 = fs["Camera.right.p1"];	p2 = fs["Camera.right.p2"];

	cvK_tmp = cv::Mat(3,3,CV_32FC1);
	cvK_tmp.at<float>(0,0) = fx;	cvK_tmp.at<float>(0,1) = 0.0f;	cvK_tmp.at<float>(0,2) = cx;
	cvK_tmp.at<float>(1,0) = 0.0f;	cvK_tmp.at<float>(1,1) = fy;	cvK_tmp.at<float>(1,2) = cy;
	cvK_tmp.at<float>(2,0) = 0.0f;	cvK_tmp.at<float>(2,1) = 0.0f;	cvK_tmp.at<float>(2,2) = 1.0f;
	
	cvD_tmp = cv::Mat(1,5,CV_32FC1);
	cvD_tmp.at<float>(0,0) = k1;
	cvD_tmp.at<float>(0,1) = k2;
	cvD_tmp.at<float>(0,2) = p1;
	cvD_tmp.at<float>(0,3) = p2;
	cvD_tmp.at<float>(0,4) = k3;

	if(cam_right_ == nullptr) 
        throw std::runtime_error("cam_right_ is not allocated.");

	cam_right_->initParams(cols, rows, cvK_tmp, cvD_tmp);

	std::cout << "fx_r: " << cam_right_->fx() <<", "
			  << "fy_r: " << cam_right_->fy() <<", "
			  << "cx_r: " << cam_right_->cx() <<", "
			  << "cy_r: " << cam_right_->cy() <<", "
			  << "cols_r: " << cam_right_->cols() <<", "
			  << "rows_r: " << cam_right_->rows() <<"\n";

	// Load user setting parameters
	// Feature tracker
	params_.feature_tracker.thres_error            = fs["feature_tracker.thres_error"];
	params_.feature_tracker.thres_bidirection      = fs["feature_tracker.thres_bidirection"];
	params_.feature_tracker.thres_sampson          = fs["feature_tracker.thres_sampson"];
	params_.feature_tracker.window_size            = (int)fs["feature_tracker.window_size"];
	params_.feature_tracker.max_level              = (int)fs["feature_tracker.max_level"];

	// Feature extractor
	params_.feature_extractor.n_features           = (int)fs["feature_extractor.n_features"];
	params_.feature_extractor.n_bins_u             = (int)fs["feature_extractor.n_bins_u"];
	params_.feature_extractor.n_bins_v             = (int)fs["feature_extractor.n_bins_v"];
	params_.feature_extractor.thres_fastscore      = fs["feature_extractor.thres_fastscore"];
	params_.feature_extractor.radius               = fs["feature_extractor.radius"];

	// Motion estimator
	params_.motion_estimator.thres_1p_error        = fs["motion_estimator.thres_1p_error"];
	params_.motion_estimator.thres_5p_error        = fs["motion_estimator.thres_5p_error"];
	params_.motion_estimator.thres_poseba_error    = fs["motion_estimator.thres_poseba_error"];

	// Keyframe update
	params_.keyframe_update.thres_alive_ratio      = fs["keyframe_update.thres_alive_ratio"];
	params_.keyframe_update.thres_mean_parallax    = fs["keyframe_update.thres_mean_parallax"];
	
	// Map update
	params_.map_update.thres_parallax = fs["map_update.thres_parallax"];
	params_.map_update.thres_parallax *= D2R;

	// Do undistortion or not.
	system_flags_.flagDoUndistortion = (int)fs["flagDoUndistortion"];

	std::cout << " - 'loadStereoCameraIntrinsicAndUserParameters()' - loaded.\n";
};

void StereoVO::saveLandmarks(const LandmarkPtrVec& lms, bool verbose){
	for(auto lm : lms)	
		all_landmarks_.push_back(lm);

	if(verbose)
		std::cout << "# of all accumulated landmarks: " << all_landmarks_.size() << std::endl;
};

void StereoVO::saveLandmark(const LandmarkPtr& lm, bool verbose){
	all_landmarks_.push_back(lm);
	
	if(verbose)
		std::cout << "# of all accumulated landmarks: " << all_landmarks_.size() << std::endl;
};

void StereoVO::saveStereoFrames(const StereoFramePtrVec& stframes, bool verbose){
	for(auto stf : stframes)
		all_stframes_.push_back(stf);
	
	if(verbose)
		std::cout << "# of all accumulated stereo frames   : " << all_stframes_.size() << std::endl;
};

void StereoVO::saveStereoFrame(const StereoFramePtr& stframe, bool verbose){
	all_stframes_.push_back(stframe);
	
	if(verbose)
		std::cout << "# of all accumulated stereo frames   : " << all_stframes_.size() << std::endl;
};

void StereoVO::saveStereoKeyframe(const StereoFramePtr& stframe, bool verbose)
{
	all_stkeyframes_.push_back(stframe);
	
	if(verbose)
		std::cout << "# of all accumulated stereo keyframes   : " << all_stkeyframes_.size() << std::endl;	
};

const cv::Mat& StereoVO::getDebugImage()
{
    return img_debug_;
};


void StereoVO::trackStereoImages(const cv::Mat& img_left, const cv::Mat& img_right, const double& timestamp)
{
    // Algorithm implementation

    
};
