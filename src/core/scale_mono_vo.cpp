#include "core/scale_mono_vo.h"

/**
 * @brief Scale mono VO object
 * @details Constructor of a scale mono VO object 
 * @param mode mode == "dataset": dataset mode, mode == "rosbag": rosbag mode. (callback based)
 * @return none
 * @author Changhyeon Kim (hyun91015@gmail.com)
 * @date 10-July-2022
 */
ScaleMonoVO::ScaleMonoVO(std::string mode, std::string directory_intrinsic)
: cam_(nullptr), system_flags_(), frame_prev_(nullptr) 
{
	std::cout << "Scale mono VO starts\n";
		
	// Initialize camera
	cam_ = std::make_shared<Camera>();

	if(mode == "dataset")
	{
		throw std::runtime_error("dataset mode is not supported.");
	}
	else if(mode == "rosbag")
	{
		std::cout << "ScaleMonoVO - 'rosbag' mode.\n";
		
		this->loadCameraIntrinsicAndUserParameters(directory_intrinsic);
		// wait callback ...
	}
	else 
		throw std::runtime_error("ScaleMonoVO - unknown mode.");

	// Initialize feature extractor (ORB-based)
	extractor_ = std::make_shared<FeatureExtractor>();
	int n_bins_u     = params_.feature_extractor.n_bins_u;
	int n_bins_v     = params_.feature_extractor.n_bins_v;
	float THRES_FAST = params_.feature_extractor.thres_fastscore;
	float radius     = params_.feature_extractor.radius;
	extractor_->initParams(cam_->cols(), cam_->rows(), n_bins_u, n_bins_v, THRES_FAST, radius);

	// Initialize feature tracker (KLT-based)
	tracker_ = std::make_shared<FeatureTracker>();

	// Initialize motion estimator
	motion_estimator_ = std::make_shared<MotionEstimator>();
	motion_estimator_->setThres1p(params_.motion_estimator.thres_1p_error);
	motion_estimator_->setThres5p(params_.motion_estimator.thres_5p_error);

	// Initialize scale estimator
	double L = 1.45; // CAR experiment.

	mut_scale_estimator_      = std::make_shared<std::mutex>();
	cond_var_scale_estimator_ = std::make_shared<std::condition_variable>(); // New pose 가 도
	flag_do_ASR_              = std::make_shared<bool>(false);
	
	scale_estimator_          = std::make_shared<ScaleEstimator>(cam_, L, mut_scale_estimator_, cond_var_scale_estimator_, flag_do_ASR_);
	
	scale_estimator_->setTurnRegion_ThresCountTurn(params_.scale_estimator.thres_cnt_turns);
	scale_estimator_->setTurnRegion_ThresPsi(params_.scale_estimator.thres_turn_psi*D2R);

	scale_estimator_->setSFP_ThresAgePastHorizon(params_.scale_estimator.thres_age_past_horizon);
	scale_estimator_->setSFP_ThresAgeUse(params_.scale_estimator.thres_age_use);
	scale_estimator_->setSFP_ThresAgeRecon(params_.scale_estimator.thres_age_recon);
	scale_estimator_->setSFP_ThresParallaxUse(params_.scale_estimator.thres_parallax_use*D2R);
	scale_estimator_->setSFP_ThresParallaxRecon(params_.scale_estimator.thres_parallax_recon*D2R);

	// Initialize keyframes class
	keyframes_ = std::make_shared<Keyframes>();
};

/**
 * @brief Scale mono VO destructor
 * @details Destructor of a scale mono VO object 
 * @return none
 * @author Changhyeon Kim (hyun91015@gmail.com)
 * @date 10-July-2022
 */
ScaleMonoVO::~ScaleMonoVO() {
	std::cout << "Scale mono VO is terminated.\n";
};

/**
 * @brief load monocular camera intrinsic parameters from yaml file.
 * @details 카메라의 intrinsic parameter (fx,fy,cx,cy, distortion)을 얻어온다. 
 * @param dir file directory
 * @return void
 * @author Changhyeon Kim (hyun91015@gmail.com)
 * @date 11-July-2022
 */
void ScaleMonoVO::loadCameraIntrinsicAndUserParameters(const std::string& dir) {
	cv::FileStorage fs(dir, cv::FileStorage::READ);
	if (!fs.isOpened()) throw std::runtime_error("intrinsic file cannot be found!\n");

	int rows, cols;
	rows = fs["Camera.height"];	cols = fs["Camera.width"];

	float fx, fy, cx, cy;
	fx = fs["Camera.fx"];	fy = fs["Camera.fy"];
	cx = fs["Camera.cx"];	cy = fs["Camera.cy"];

	float k1,k2,k3,p1,p2;
	k1 = fs["Camera.k1"];	k2 = fs["Camera.k2"];	k3 = fs["Camera.k3"];
	p1 = fs["Camera.p1"];	p2 = fs["Camera.p2"];

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

	if(cam_ == nullptr) throw std::runtime_error("cam_ is not allocated.");
	cam_->initParams(cols, rows, cvK_tmp, cvD_tmp);

	std::cout << "fx: " << cam_->fx() <<", "
			  << "fy: " << cam_->fy() <<", "
			  << "cx: " << cam_->cx() <<", "
			  << "cy: " << cam_->cy() <<", "
			  << "cols: " << cam_->cols() <<", "
			  << "rows: " << cam_->rows() <<"\n";

	// Load user setting parameters
	// Feature tracker
	params_.feature_tracker.thres_error = fs["feature_tracker.thres_error"];
	params_.feature_tracker.thres_bidirection = fs["feature_tracker.thres_bidirection"];
	params_.feature_tracker.thres_sampson = fs["feature_tracker.thres_sampson"];
	params_.feature_tracker.window_size = (int)fs["feature_tracker.window_size"];
	params_.feature_tracker.max_level = (int)fs["feature_tracker.max_level"];

	// Feature extractor
	params_.feature_extractor.n_features = (int)fs["feature_extractor.n_features"];
	params_.feature_extractor.n_bins_u   = (int)fs["feature_extractor.n_bins_u"];
	params_.feature_extractor.n_bins_v   = (int)fs["feature_extractor.n_bins_v"];
	params_.feature_extractor.thres_fastscore = fs["feature_extractor.thres_fastscore"];
	params_.feature_extractor.radius          = fs["feature_extractor.radius"];

	// Motion estimator
	params_.motion_estimator.thres_1p_error = fs["motion_estimator.thres_1p_error"];
	params_.motion_estimator.thres_5p_error = fs["motion_estimator.thres_5p_error"];
	params_.motion_estimator.thres_poseba_error = fs["motion_estimator.thres_poseba_error"];

	// Scale estimator
	params_.scale_estimator.initial_scale          = fs["scale_estimator.initial_scale"];
	params_.scale_estimator.thres_turn_psi         = fs["scale_estimator.thres_turn_psi"];
	params_.scale_estimator.thres_cnt_turns        = (int)fs["scale_estimator.thres_cnt_turns"];
	params_.scale_estimator.thres_age_past_horizon = (int)fs["scale_estimator.thres_age_past_horizon"];
	params_.scale_estimator.thres_age_use          = (int)fs["scale_estimator.thres_age_use"];
	params_.scale_estimator.thres_age_recon        = (int)fs["scale_estimator.thres_age_recon"];
	params_.scale_estimator.thres_parallax_use     = fs["scale_estimator.thres_parallax_use"];
	params_.scale_estimator.thres_parallax_recon   = fs["scale_estimator.thres_parallax_recon"];

	// Keyframe update
	params_.keyframe_update.thres_alive_ratio   = fs["keyframe_update.thres_alive_ratio"];
	params_.keyframe_update.thres_mean_parallax = fs["keyframe_update.thres_mean_parallax"];
	
	// Map update
	params_.map_update.thres_parallax = fs["map_update.thres_parallax"];
	params_.map_update.thres_parallax *= D2R;

	// Do undistortion or not.
	system_flags_.flagDoUndistortion = (int)fs["flagDoUndistortion"];

	std::cout << " - 'loadCameraIntrinsic()' - loaded.\n";
};


/**
 * @brief Prune out invalid landmarks and their trackings.
 * @details Prune out invalid landmarks and their trackings.
 * @return num_of_valid_points
 * @author Changhyeon Kim (hyun91015@gmail.com)
 * @date 06-August-2022
 */
int ScaleMonoVO::pruneInvalidLandmarks(const PixelVec& pts0, const PixelVec& pts1, const LandmarkPtrVec& lms, const MaskVec& mask,
	PixelVec& pts0_alive, PixelVec& pts1_alive, LandmarkPtrVec& lms_alive)
{
	if(pts0.size() != pts1.size() || pts0.size() != lms.size())
		throw std::runtime_error("pts0.size() != pts1.size() || pts0.size() != lms.size()");
	
	int n_pts = pts0.size();

	// Tracking 결과를 반영하여 pts1_alive, lms1_alive를 정리한다.
	pts0_alive.resize(0);
	pts1_alive.resize(0);
	lms_alive.resize(0);
	pts0_alive.reserve(n_pts);
	pts1_alive.reserve(n_pts);
	lms_alive.reserve(n_pts);
	int cnt_alive = 0;
	for(int i = 0; i < n_pts; ++i){
		if( mask[i] ) {
			pts0_alive.push_back(pts0[i]);
			pts1_alive.push_back(pts1[i]);
			lms_alive.push_back(lms[i]);
			++cnt_alive;
		}
		else lms[i]->setDead(); // track failed. Dead point.
	}
	return cnt_alive;
};
/**
 * @brief Prune out invalid landmarks and their trackings.
 * @details Prune out invalid landmarks and their trackings.
 * @return num_of_valid_points
 * @author Changhyeon Kim (hyun91015@gmail.com)
 * @date 06-August-2022
 */
int ScaleMonoVO::pruneInvalidLandmarks(const LandmarkTracking& lmtrack, const MaskVec& mask,
	LandmarkTracking& lmtrack_alive)
{
	if(lmtrack.pts0.size() != lmtrack.pts1.size() || lmtrack.pts0.size() != lmtrack.lms.size())
		throw std::runtime_error("lmtrack.pts0.size() != lmtrack.pts1.size() || lmtrack.pts0.size() != lmtrack.lms.size()");
	
	int n_pts = lmtrack.pts0.size();

	// Tracking 결과를 반영하여 pts1_alive, lms1_alive를 정리한다.
	lmtrack_alive.pts0.resize(0);
	lmtrack_alive.scale_change.resize(0);
	lmtrack_alive.pts1.resize(0);
	lmtrack_alive.lms.resize(0);
	lmtrack_alive.pts0.reserve(n_pts);
	lmtrack_alive.scale_change.reserve(n_pts);
	lmtrack_alive.pts1.reserve(n_pts);
	lmtrack_alive.lms.reserve(n_pts);
	int cnt_alive = 0;
	for(int i = 0; i < n_pts; ++i){
		if( mask[i]) {
			lmtrack_alive.pts0.push_back(lmtrack.pts0[i]);
			lmtrack_alive.pts1.push_back(lmtrack.pts1[i]);
			lmtrack_alive.scale_change.push_back(lmtrack.scale_change[i]);
			lmtrack_alive.lms.push_back(lmtrack.lms[i]);
			++cnt_alive;
		}
		else lmtrack.lms[i]->setDead(); // track failed. Dead point.
	}
	return cnt_alive;
};

/**
 * @brief Update keyframe with an input frame
 * @details 새로운 frame으로 Keyframe을 업데이트 & all_keyframes_ 에 저장
 * @param frame Keyframe이 될 frame
 * @return void
 * @author Changhyeon Kim (hyun91015@gmail.com)
 * @date 12-July-2022
 */
void ScaleMonoVO::updateKeyframe(const FramePtr& frame){
};

void ScaleMonoVO::saveLandmarks(const LandmarkPtrVec& lms, bool verbose){
	for(auto lm : lms)	
		all_landmarks_.push_back(lm);

	if(verbose)
		std::cout << "# of all accumulated landmarks: " << all_landmarks_.size() << std::endl;
};

void ScaleMonoVO::saveLandmark(const LandmarkPtr& lm, bool verbose){
	all_landmarks_.push_back(lm);
	
	if(verbose)
		std::cout << "# of all accumulated landmarks: " << all_landmarks_.size() << std::endl;
};

void ScaleMonoVO::saveFrames(const FramePtrVec& frames, bool verbose){
	for(auto f : frames)
		all_frames_.push_back(f);
	
	if(verbose)
		std::cout << "# of all accumulated frames   : " << all_frames_.size() << std::endl;
};

void ScaleMonoVO::saveFrame(const FramePtr& frame, bool verbose){
	all_frames_.push_back(frame);
	
	if(verbose)
		std::cout << "# of all accumulated frames   : " << all_frames_.size() << std::endl;
};

float ScaleMonoVO::calcLandmarksMeanAge(const LandmarkPtrVec& lms){
	float mean_age = 0.0f;
	float n_lms = lms.size();

	for(int i = 0; i < n_lms; ++i){
		mean_age += lms[i]->getAge();
	}
	mean_age /= n_lms;

	return mean_age;
};

void ScaleMonoVO::showTracking(const std::string& window_name, const cv::Mat& img, const PixelVec& pts0, const PixelVec& pts1, const PixelVec& pts1_new){
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
	cv::waitKey(3);
};

void ScaleMonoVO::showTrackingBA(const std::string& window_name, const cv::Mat& img, const PixelVec& pts1, const PixelVec& pts1_project, const MaskVec& mask_valid){	
	cv::namedWindow(window_name);
	img.copyTo(img_debug_);
	cv::cvtColor(img_debug_, img_debug_, CV_GRAY2RGB);
	for(int i = 0; i < pts1.size(); ++i) {
		if(mask_valid[i])
			cv::circle(img_debug_, pts1[i], 1.0, cv::Scalar(0,0,255),4); // alived magenta
	}
	for(int i = 0; i < pts1.size(); ++i){
		if(mask_valid[i])
			cv::rectangle(img_debug_, cv::Point2f(pts1_project[i].x-8,pts1_project[i].y-8),cv::Point2f(pts1_project[i].x+4,pts1_project[i].y+4), 
				cv::Scalar(0,255,0), 1);
	}
	
	cv::imshow(window_name, img_debug_);
	cv::waitKey(3);
};

void ScaleMonoVO::showTracking(const std::string& window_name, const cv::Mat& img, const LandmarkPtrVec& lms){
	cv::namedWindow(window_name);
	img.copyTo(img_debug_);
	cv::cvtColor(img_debug_, img_debug_, CV_GRAY2RGB);

	for(int i = 0; i < lms.size(); ++i){
		const LandmarkPtr& lm = lms[i];
		const PixelVec& pts = lm->getObservations();
		// std::cout <<"track size: " << pts.size() <<std::endl;
		uint32_t n_past = pts.size();
		if(n_past > 5) n_past = 5;
		for(int j = pts.size()-1; j >= pts.size()-n_past+1; --j){
			const Pixel& p1 = pts[j];
			const Pixel& p0 = pts[j-1];
			cv::line(img_debug_, p0,p1, cv::Scalar(0,255,255),1);
			cv::circle(img_debug_, p0, 3.0, cv::Scalar(0,0,0),2); // green tracked points
			cv::circle(img_debug_, p0, 2.0, cv::Scalar(0,255,0),1); // green tracked points
			cv::circle(img_debug_, p1, 3.0, cv::Scalar(0,0,0),2); // green tracked points
			cv::circle(img_debug_, p1, 2.0, cv::Scalar(0,255,0),1); // green tracked points
		}
	}
	cv::imshow(window_name, img_debug_);
	cv::waitKey(3);
};


ScaleMonoVO::AlgorithmStatistics ScaleMonoVO::getStatistics() const{
	return stat_;
};

const cv::Mat& ScaleMonoVO::getDebugImage()
{
	return img_debug_;
};