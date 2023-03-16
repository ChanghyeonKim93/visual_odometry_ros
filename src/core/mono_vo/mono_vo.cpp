#include "core/mono_vo/mono_vo.h"

/**
 * @brief Mono VO object
 * @details Constructor of a Mono VO object 
 * @param mode mode == "dataset": dataset mode, mode == "rosbag": rosbag mode. (callback based)
 * @return none
 * @author Changhyeon Kim (hyun91015@gmail.com)
 * @date 10-July-2022
 */
MonoVO::MonoVO(std::string mode, std::string directory_intrinsic)
: cam_(nullptr), system_flags_(), frame_prev_(nullptr), keyframe_ref_(nullptr)
{
	std::cout << "Mono VO starts\n";
		
	// Initialize camera
	cam_ = std::make_shared<Camera>();

	if(mode == "dataset")
	{
		throw std::runtime_error("dataset mode is not supported.");
	}
	else if(mode == "rosbag")
	{
		std::cout << "MonoVO - 'rosbag' mode.\n";
		
		this->loadCameraIntrinsicAndUserParameters(directory_intrinsic);
		// wait callback ...
	}
	else 
		throw std::runtime_error("MonoVO - unknown mode.");

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

	// Initialize keyframes class
	keyframes_ = std::make_shared<Keyframes>();
	keyframes_->setMaxKeyframes(params_.keyframe_update.n_max_keyframes_in_window);
	keyframes_->setThresOverlapRatio(params_.keyframe_update.thres_overlap_ratio);
	keyframes_->setThresTranslation(params_.keyframe_update.thres_translation);
	keyframes_->setThresRotation(params_.keyframe_update.thres_rotation*D2R);
};

/**
 * @brief Scale mono VO destructor
 * @details Destructor of a scale mono VO object 
 * @return none
 * @author Changhyeon Kim (hyun91015@gmail.com)
 * @date 10-July-2022
 */
MonoVO::~MonoVO() 
{
	std::cout << "Save all frames trajectory...\n";
	std::string filedir_frame_poses = "/home/kch/frame_poses.txt";
	std::ofstream of_frame_poses(filedir_frame_poses, std::ios::trunc);
    of_frame_poses.precision(4);
    of_frame_poses.setf(std::ios_base::fixed, std::ios_base::floatfield);
    if(of_frame_poses.is_open())
	{
		for(int j = 0; j < this->all_frames_.size(); ++j)
		{
			of_frame_poses  << all_frames_[j]->getID() << " "
							<< all_frames_[j]->getPose()(0,0) << " " 
							<< all_frames_[j]->getPose()(0,1) << " " 
							<< all_frames_[j]->getPose()(0,2) << " " 
							<< all_frames_[j]->getPose()(0,3) << " " 
							<< all_frames_[j]->getPose()(1,0) << " " 
							<< all_frames_[j]->getPose()(1,1) << " " 
							<< all_frames_[j]->getPose()(1,2) << " " 
							<< all_frames_[j]->getPose()(1,3) << " " 
							<< all_frames_[j]->getPose()(2,0) << " " 
							<< all_frames_[j]->getPose()(2,1) << " " 
							<< all_frames_[j]->getPose()(2,2) << " " 
							<< all_frames_[j]->getPose()(2,3) << std::endl;
		}
    }
    else {
        throw std::runtime_error("file_dir cannot be opened!");
    }
	std::cout << " DONE!\n";

	std::cout << "Save all keyframes trajectory...\n";
	std::string filedir_keyframe_poses = "/home/kch/keyframe_poses.txt";
	std::ofstream of_keyframe_poses(filedir_keyframe_poses, std::ios::trunc);
    of_keyframe_poses.precision(4);
    of_keyframe_poses.setf(std::ios_base::fixed, std::ios_base::floatfield);
    if(of_keyframe_poses.is_open())
	{
		for(int j = 0; j < this->all_keyframes_.size(); ++j)
		{
			of_keyframe_poses   << all_keyframes_[j]->getID() << " "
								<< all_keyframes_[j]->getPose()(0,0) << " " 
								<< all_keyframes_[j]->getPose()(0,1) << " " 
								<< all_keyframes_[j]->getPose()(0,2) << " " 
								<< all_keyframes_[j]->getPose()(0,3) << " " 
								<< all_keyframes_[j]->getPose()(1,0) << " " 
								<< all_keyframes_[j]->getPose()(1,1) << " " 
								<< all_keyframes_[j]->getPose()(1,2) << " " 
								<< all_keyframes_[j]->getPose()(1,3) << " " 
								<< all_keyframes_[j]->getPose()(2,0) << " " 
								<< all_keyframes_[j]->getPose()(2,1) << " " 
								<< all_keyframes_[j]->getPose()(2,2) << " " 
								<< all_keyframes_[j]->getPose()(2,3) << std::endl;
		}
    }
    else {
        throw std::runtime_error("file_dir cannot be opened!");
    }
	std::cout << " DONE!\n";

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
void MonoVO::loadCameraIntrinsicAndUserParameters(const std::string& dir) {
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
	params_.keyframe_update.thres_translation         = fs["keyframe_update.thres_translation"];
	params_.keyframe_update.thres_rotation            = fs["keyframe_update.thres_rotation"]; 
	params_.keyframe_update.thres_overlap_ratio       = fs["keyframe_update.thres_overlap_ratio"];
	params_.keyframe_update.n_max_keyframes_in_window = fs["keyframe_update.n_max_keyframes_in_window"];
	
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
int MonoVO::pruneInvalidLandmarks(const PixelVec& pts0, const PixelVec& pts1, const LandmarkPtrVec& lms, const MaskVec& mask,
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
		else 
			lms[i]->setUntracked(); // track failed. Dead point.
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
int MonoVO::pruneInvalidLandmarks(const LandmarkTracking& lmtrack, const MaskVec& mask,
	LandmarkTracking& lmtrack_alive)
{
	if(lmtrack.pts0.size() != lmtrack.pts1.size() || lmtrack.pts0.size() != lmtrack.lms.size())
		throw std::runtime_error("lmtrack.pts0.size() != lmtrack.pts1.size() || lmtrack.pts0.size() != lmtrack.lms.size()");
	
	int n_pts = lmtrack.pts0.size();

	// Tracking 결과를 반영하여 pts1_alive, lms1_alive를 정리한다.
	// lmtrack_alive.pts0.resize(0);
	// lmtrack_alive.scale_change.resize(0);
	// lmtrack_alive.pts1.resize(0);
	// lmtrack_alive.lms.resize(0);
	// lmtrack_alive.pts0.reserve(n_pts);
	// lmtrack_alive.scale_change.reserve(n_pts);
	// lmtrack_alive.pts1.reserve(n_pts);
	// lmtrack_alive.lms.reserve(n_pts);
	// int cnt_alive = 0;
	// for(int i = 0; i < n_pts; ++i)
	// {
	// 	if( mask[i]) {
	// 		lmtrack_alive.pts0.push_back(lmtrack.pts0[i]);
	// 		lmtrack_alive.pts1.push_back(lmtrack.pts1[i]);
	// 		lmtrack_alive.scale_change.push_back(lmtrack.scale_change[i]);
	// 		lmtrack_alive.lms.push_back(lmtrack.lms[i]);
	// 		++cnt_alive;
	// 	}
	// 	else lmtrack.lms[i]->setDead(); // track failed. Dead point.
	// }

	std::vector<int> index_valid;
	index_valid.reserve(n_pts);
	int cnt_alive = 0;
	for(int i = 0; i < n_pts; ++i)
	{
		if( mask[i] && lmtrack.lms[i]->isAlive() )
		{
			index_valid.push_back(i);
			++cnt_alive;
		}
		else
			lmtrack.lms[i]->setUntracked();
	}

	lmtrack_alive.pts0.resize(cnt_alive);
	lmtrack_alive.scale_change.resize(cnt_alive);
	lmtrack_alive.pts1.resize(cnt_alive);
	lmtrack_alive.lms.resize(cnt_alive);
	for(int i = 0; i < cnt_alive; ++i)
	{
		const int& idx = index_valid[i];
		lmtrack_alive.pts0[i] = lmtrack.pts0[idx];
		lmtrack_alive.pts1[i] = lmtrack.pts1[idx];
		lmtrack_alive.scale_change[i] = lmtrack.scale_change[idx];
		lmtrack_alive.lms[i] = lmtrack.lms[idx];
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
void MonoVO::updateKeyframe(const FramePtr& frame){
};

void MonoVO::saveLandmarks(const LandmarkPtrVec& lms, bool verbose){
	for(auto lm : lms)	
		all_landmarks_.push_back(lm);

	if(verbose)
		std::cout << "# of all accumulated landmarks: " << all_landmarks_.size() << std::endl;
};

void MonoVO::saveLandmark(const LandmarkPtr& lm, bool verbose){
	all_landmarks_.push_back(lm);
	
	if(verbose)
		std::cout << "# of all accumulated landmarks: " << all_landmarks_.size() << std::endl;
};

void MonoVO::saveFrames(const FramePtrVec& frames, bool verbose){
	for(auto f : frames)
		all_frames_.push_back(f);
	
	if(verbose)
		std::cout << "# of all accumulated frames   : " << all_frames_.size() << std::endl;
};

void MonoVO::saveFrame(const FramePtr& frame, bool verbose){
	all_frames_.push_back(frame);
	
	if(verbose)
		std::cout << "# of all accumulated frames   : " << all_frames_.size() << std::endl;
};

void MonoVO::saveKeyframe(const FramePtr& frame, bool verbose)
{
	all_keyframes_.push_back(frame);
	
	if(verbose)
		std::cout << "# of all accumulated keyframes   : " << all_keyframes_.size() << std::endl;	
};

float MonoVO::calcLandmarksMeanAge(const LandmarkPtrVec& lms){
	float mean_age = 0.0f;
	float n_lms = lms.size();

	for(int i = 0; i < n_lms; ++i){
		mean_age += lms[i]->getAge();
	}
	mean_age /= n_lms;

	return mean_age;
};

void MonoVO::showTracking(const std::string& window_name, const cv::Mat& img, const PixelVec& pts0, const PixelVec& pts1, const PixelVec& pts1_new){
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

void MonoVO::showTrackingBA(const std::string& window_name, const cv::Mat& img, const PixelVec& pts1, const PixelVec& pts1_project){	
	int rect_half = 6;
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
		cv::rectangle(img_debug_, 
			cv::Point2f(pts1_project[i].x-rect_half,pts1_project[i].y-rect_half),
			cv::Point2f(pts1_project[i].x+rect_half,pts1_project[i].y+rect_half), 
			color_green, 2);
	}
	
	cv::imshow(window_name, img_debug_);
	cv::waitKey(2);
};

void MonoVO::showTracking(const std::string& window_name, const cv::Mat& img, const LandmarkPtrVec& lms){
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


const MonoVO::AlgorithmStatistics& MonoVO::getStatistics() const{
	return stat_;
};

const cv::Mat& MonoVO::getDebugImage()
{
	return img_debug_;
};