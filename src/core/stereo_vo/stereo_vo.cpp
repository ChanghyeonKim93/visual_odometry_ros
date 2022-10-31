#include "core/stereo_vo/stereo_vo.h"

StereoVO::StereoVO(std::string mode, std::string directory_intrinsic)
: stereo_cam_(nullptr), stframe_prev_(nullptr)
{
	std::cout << "Stereo VO starts\n";
	
	// Set the algorithm to 'stereo mode'
	bool is_stereo_mode = true;

	// Initialize stereo camera
	stereo_cam_ = std::make_shared<StereoCamera>();
	
	if(mode == "dataset")
	{
		throw std::runtime_error("StereoVO - 'dataset' mode is not supported now...");
	}
	else if(mode == "rosbag")
	{
		std::cout << "StereoVO - 'rosbag' mode.\n";
		this->loadStereoCameraIntrinsicAndUserParameters(directory_intrinsic);
	}
	else 
		throw std::runtime_error("StereoVO - unknown mode...");

	// Get rectified stereo camera parameters
	CameraConstPtr& cam_rect = stereo_cam_->getRectifiedCamera();
	const PoseSE3& T_lr = stereo_cam_->getRectifiedStereoPoseLeft2Right(); // left to right pose (rectified camera)

	// Initialize feature extractor (ORB-based)
	int n_bins_u     = params_.feature_extractor.n_bins_u;
	int n_bins_v     = params_.feature_extractor.n_bins_v;
	float THRES_FAST = params_.feature_extractor.thres_fastscore;
	float radius     = params_.feature_extractor.radius;

	extractor_ = std::make_shared<FeatureExtractor>();
	extractor_->initParams(cam_rect->cols(), cam_rect->rows(), n_bins_u, n_bins_v, THRES_FAST, radius);

	// Initialize feature tracker (KLT-based)
	tracker_ = std::make_shared<FeatureTracker>();

	// Initialize motion estimator
	motion_estimator_ = std::make_shared<MotionEstimator>(is_stereo_mode, T_lr);
	motion_estimator_->setThres1p(params_.motion_estimator.thres_1p_error);
	motion_estimator_->setThres5p(params_.motion_estimator.thres_5p_error);

	// Initialize stereo keyframes class
	stkeyframes_ = std::make_shared<StereoKeyframes>();
	stkeyframes_->setMaxStereoKeyframes(params_.keyframe_update.n_max_keyframes_in_window);
	stkeyframes_->setThresOverlapRatio(params_.keyframe_update.thres_alive_ratio);
	stkeyframes_->setThresRotation(params_.keyframe_update.thres_rotation*D2R);
	stkeyframes_->setThresTranslation(params_.keyframe_update.thres_trans);
};

StereoVO::~StereoVO()
{

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

	if(stereo_cam_->getLeftCamera() == nullptr) 
        throw std::runtime_error("cam_left_ is not allocated.");

	stereo_cam_->getLeftCamera()->initParams(cols, rows, cvK_tmp, cvD_tmp);

	std::cout <<"LEFT  CAMERA PARAMETERS:\n";
	std::cout << "fx_l: " << stereo_cam_->getLeftCamera()->fx() <<", "
			  << "fy_l: " << stereo_cam_->getLeftCamera()->fy() <<", "
			  << "cx_l: " << stereo_cam_->getLeftCamera()->cx() <<", "
			  << "cy_l: " << stereo_cam_->getLeftCamera()->cy() <<", "
			  << "cols_l: " << stereo_cam_->getLeftCamera()->cols() <<", "
			  << "rows_l: " << stereo_cam_->getLeftCamera()->rows() <<"\n";
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

	if(stereo_cam_->getRightCamera() == nullptr) 
        throw std::runtime_error("cam_right_ is not allocated.");

	stereo_cam_->getRightCamera()->initParams(cols, rows, cvK_tmp, cvD_tmp);

	std::cout <<"RIGHT CAMERA PARAMETERS:\n";
	std::cout << "fx_r: " << stereo_cam_->getRightCamera()->fx() <<", "
			  << "fy_r: " << stereo_cam_->getRightCamera()->fy() <<", "
			  << "cx_r: " << stereo_cam_->getRightCamera()->cx() <<", "
			  << "cy_r: " << stereo_cam_->getRightCamera()->cy() <<", "
			  << "cols_r: " << stereo_cam_->getRightCamera()->cols() <<", "
			  << "rows_r: " << stereo_cam_->getRightCamera()->rows() <<"\n";


	cv::Mat cvT_lr_tmp = cv::Mat(4,4,CV_32FC1);
	PoseSE3 T_lr;
    fs["T_lr"] >> cvT_lr_tmp;
	for(int i = 0; i < 4; ++i)
		for(int j = 0; j < 4; ++j)
			T_lr(i,j) = cvT_lr_tmp.at<float>(i,j);

	stereo_cam_->setStereoPoseLeft2Right(T_lr);

	std::cout <<"Stereo pose (left to right) T_lr:\n" << T_lr << std::endl;

	stereo_cam_->initStereoCameraToRectify();
	
	// Do undistortion or not.
	system_flags_.flagDoUndistortion = (int)fs["flagDoUndistortion"];

	// Load user setting parameters
	// Feature tracker
	params_.feature_tracker.thres_error            = fs["feature_tracker.thres_error"];
	params_.feature_tracker.thres_bidirection      = fs["feature_tracker.thres_bidirection"];
	params_.feature_tracker.thres_sampson          = fs["feature_tracker.thres_sampson"];
	params_.feature_tracker.window_size            = (int)fs["feature_tracker.window_size"];
	params_.feature_tracker.max_level              = (int)fs["feature_tracker.max_level"];

	// Map update
	params_.map_update.thres_parallax              = fs["map_update.thres_parallax"];
	params_.map_update.thres_parallax *= D2R;

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
	params_.keyframe_update.thres_trans            = fs["keyframe_update.thres_trans"];
	params_.keyframe_update.thres_rotation         = fs["keyframe_update.thres_rotation"];
	params_.keyframe_update.n_max_keyframes_in_window = (int)fs["keyframe_update.n_max_keyframes_in_window"];

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

const StereoVO::AlgorithmStatistics& StereoVO::getStatistics() const{
	return stat_;
};

const cv::Mat& StereoVO::getDebugImage()
{
    return img_debug_;
};


void StereoVO::showTracking(const std::string& window_name, const cv::Mat& img, const PixelVec& pts0, const PixelVec& pts1, const PixelVec& pts1_new)
{
	cv::namedWindow(window_name);
	img.copyTo(img_debug_);
	cv::cvtColor(img_debug_, img_debug_, CV_GRAY2RGB);
	for(int i = 0; i < pts1.size(); ++i){
		cv::line(img_debug_,pts0[i],pts1[i], cv::Scalar(0,255,255),1);
	}
	for(int i = 0; i < pts0.size(); ++i) {
		cv::circle(img_debug_, pts0[i], 3.0, cv::Scalar(0,0,0),2); // alived magenta
		cv::circle(img_debug_, pts0[i], 2.0, cv::Scalar(255,0,255),1); // alived magenta
	}
	for(int i = 0; i < pts1.size(); ++i){
		cv::circle(img_debug_, pts1[i], 3.0, cv::Scalar(0,0,0),2); // green tracked points
		cv::circle(img_debug_, pts1[i], 2.0, cv::Scalar(0,255,0),1); // green tracked points
	}
	for(int i = 0; i < pts1_new.size(); ++i){
		cv::circle(img_debug_, pts1_new[i], 3.0, cv::Scalar(0,0,0), 2); // blue new points
		cv::circle(img_debug_, pts1_new[i], 2.0, cv::Scalar(255,0,0),1); // blue new points
	}
	
	cv::imshow(window_name, img_debug_);
	cv::waitKey(2);
};

void StereoVO::showTrackingBA(const std::string& window_name, const cv::Mat& img, const PixelVec& pts1, const PixelVec& pts1_project)
{
	int rect_half     = 6;
	int circle_radius = 4;

	cv::Scalar color_red   = cv::Scalar(0,0,255);
	cv::Scalar color_green = cv::Scalar(0,255,0);

	cv::namedWindow(window_name);
	img.copyTo(img_debug_);
	cv::cvtColor(img_debug_, img_debug_, CV_GRAY2RGB);
	for(int i = 0; i < pts1.size(); ++i) {
		cv::circle(img_debug_, pts1[i], 1.0, color_red, circle_radius); // alived magenta
	}
	for(int i = 0; i < pts1_project.size(); ++i){
		cv::rectangle(img_debug_, cv::Point2f(pts1_project[i].x-rect_half,pts1_project[i].y-rect_half),cv::Point2f(pts1_project[i].x+rect_half,pts1_project[i].y+rect_half), 
			color_green, 2);
	}
	
	cv::imshow(window_name, img_debug_);
	cv::waitKey(2);
};

