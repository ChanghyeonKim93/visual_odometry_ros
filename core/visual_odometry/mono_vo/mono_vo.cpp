#include "core/visual_odometry/mono_vo/mono_vo.h"

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



/**
 * @brief function to track a new image (local bundle mode)
 * @details 새로 들어온 이미지의 자세를 구하는 함수. 만약, mono vo가 초기화되지 않은 경우, 해당 이미지를 초기 이미지로 설정. 
 * @param img 입력 이미지 (CV_8UC1)
 * @param timestamp 입력 이미지의 timestamp.
 * @return none
 * @author Changhyeon Kim (hyun91015@gmail.com)
 * @date 10-July-2022
 */
void MonoVO::trackImage(const cv::Mat& img, const double& timestamp)
{
	float THRES_SAMPSON  = params_.feature_tracker.thres_sampson;
	float THRES_PARALLAX = params_.map_update.thres_parallax;

	// Generate statistics
	AlgorithmStatistics::LandmarkStatistics  statcurr_landmark;
	AlgorithmStatistics::FrameStatistics     statcurr_frame;
	AlgorithmStatistics::KeyframeStatistics  statcurr_keyframe;
	AlgorithmStatistics::ExecutionStatistics statcurr_execution;
			
	// 이미지 undistort (KITTI라서 할 필요 X)
	cv::Mat img_undist;
	if(system_flags_.flagDoUndistortion)
	{
		cam_->undistortImage(img, img_undist);
		img_undist.convertTo(img_undist, CV_8UC1);
	}
	else 
		img.copyTo(img_undist);

	// 현재 이미지에 대한 새로운 Frame 생성
	FramePtr frame_curr = std::make_shared<Frame>(cam_, img_undist, timestamp, false, nullptr);
	this->saveFrame(frame_curr);

	// Get previous and current images
	const cv::Mat& I0 = frame_prev_->getImage();
	const cv::Mat& I1 = frame_curr->getImage();

	if( !system_flags_.flagVOInit ) 
	{ 
		// 초기화 미완료
		if( !system_flags_.flagFirstImageGot )
		{ 
			// 최초 이미지
			LandmarkTracking lmtrack_curr;

			// Extract pixels
			extractor_->resetWeightBin();
			extractor_->extractORBwithBinning_fast(I1, lmtrack_curr.pts1, true);

			// 초기 landmark 생성
			for(const auto& pt : lmtrack_curr.pts1)
			{
				LandmarkPtr lm_new = std::make_shared<Landmark>(pt, frame_curr, cam_);
				lmtrack_curr.lms.push_back(lm_new);
			}
			
			// Related Landmark와 tracked pixels를 업데이트
			frame_curr->setPtsSeenAndRelatedLandmarks(lmtrack_curr.pts1, lmtrack_curr.lms);
		
			PoseSE3 T_init = PoseSE3::Identity();
			T_init.block<3,1>(0,3) << 0,0,-1; // get initial scale.
			frame_curr->setPose(PoseSE3::Identity());
			frame_curr->setPoseDiff10(T_init);
			
			this->saveLandmarks(lmtrack_curr.lms); // save all newly detected landmarks

			if( true )
				this->showTracking("img_features", I1, lmtrack_curr.pts1, PixelVec(), PixelVec());

			std::cout << "First image is initialized.\n";

			// 첫 이미지 업데이트 완료
			system_flags_.flagFirstImageGot = true;
		}
		else 
		{
			// Get previously tracked landmarks
			LandmarkTracking lmtrack_prev(frame_prev_->getPtsSeen(), frame_prev_->getPtsSeen(), frame_prev_->getRelatedLandmarkPtr());
	
			// Get previous pose differences. We assume that constant velocity model.
			PoseSE3 Twc_prev   = frame_prev_->getPose();
			PoseSE3 dT01_prior = frame_prev_->getPoseDiff01();

			// 'frame_prev_' 의 lms 를 현재 이미지로 track. 5ms
			MaskVec mask_track;
			tracker_->track(I0, I1, lmtrack_prev.pts0, 
				params_.feature_tracker.window_size, params_.feature_tracker.max_level, params_.feature_tracker.thres_error,
				lmtrack_prev.pts1, mask_track);

			LandmarkTracking lmtrack_klt(lmtrack_prev, mask_track);

			// Scale refinement 50ms
			MaskVec mask_refine(lmtrack_klt.n_pts, true);			
			LandmarkTracking lmtrack_scaleok(lmtrack_klt, mask_refine);

			// 5-point algorithm 2ms
			MaskVec  mask_5p(lmtrack_scaleok.n_pts);
			PointVec X0_inlier(lmtrack_scaleok.n_pts);
			
			Rot3 dR10; Pos3 dt10;
			if( !motion_estimator_->calcPose5PointsAlgorithm(lmtrack_scaleok.pts0, lmtrack_scaleok.pts1, cam_, dR10, dt10, X0_inlier, mask_5p) ) 
				throw std::runtime_error("calcPose5PointsAlgorithm() is failed.");

			// Check sampson distance 0.01 ms
			std::vector<float> symm_epi_dist;
			motion_estimator_->calcSampsonDistance(lmtrack_scaleok.pts0, lmtrack_scaleok.pts1, cam_, dR10, dt10, symm_epi_dist);
			MaskVec mask_sampson(lmtrack_scaleok.n_pts);
			for(int i = 0; i < mask_sampson.size(); ++i)
				mask_sampson[i] = mask_5p[i] && (symm_epi_dist[i] < THRES_SAMPSON);
			
			LandmarkTracking lmtrack_final(lmtrack_scaleok, mask_sampson);

			// Update tracking results
			for(int i = 0; i < lmtrack_final.n_pts; ++i)
				lmtrack_final.lms[i]->addObservationAndRelatedFrame(lmtrack_final.pts1[i], frame_curr);
			
			// Frame_curr의 자세를 넣는다.
			dt10 = dt10/dt10.norm()*1.0f;
			PoseSE3 dT10; dT10 << dR10, dt10, 0.0f, 0.0f, 0.0f, 1.0f;
			PoseSE3 dT01 = geometry::inverseSE3_f(dT10);

			frame_curr->setPose(Twc_prev*dT01);
			frame_curr->setPoseDiff10(dT10);
							
#ifdef RECORD_FRAME_STAT
statcurr_frame.Twc   = frame_curr->getPose();
statcurr_frame.Tcw   = frame_curr->getPoseInv();
statcurr_frame.dT_10 = frame_curr->getPoseDiff10();
statcurr_frame.dT_01 = frame_curr->getPoseDiff01();
#endif
			
			// 빈 곳에 특징점 pts1_new 를 추출한다. 2 ms
			PixelVec pts1_new;
			extractor_->updateWeightBin(lmtrack_final.pts1); // 이미 pts1가 있는 곳은 제외.
			extractor_->extractORBwithBinning_fast(I1, pts1_new, true);

			if( true )
				this->showTracking("img_features", I1, lmtrack_final.pts0, lmtrack_final.pts1, pts1_new);
			
			if( pts1_new.size() > 0 )
			{
				// 새로운 특징점을 back-track.
				PixelVec pts0_new;
				MaskVec mask_new;
				tracker_->trackBidirection(I1, I0, pts1_new,
					params_.feature_tracker.window_size, params_.feature_tracker.max_level, params_.feature_tracker.thres_error, params_.feature_tracker.thres_bidirection,
					pts0_new, mask_new);

				// 새로운 특징점은 새로운 landmark가 된다.
				for(int i = 0; i < pts1_new.size(); ++i) 
				{
					if( mask_new[i] )
					{
						const Pixel& p0_new = pts0_new[i];
						const Pixel& p1_new = pts1_new[i];
						
						LandmarkPtr lmptr = std::make_shared<Landmark>(p0_new, frame_prev_, cam_);
						lmptr->addObservationAndRelatedFrame(p1_new, frame_curr);

						lmtrack_final.pts0.push_back(p0_new);
						lmtrack_final.pts1.push_back(p1_new);
						lmtrack_final.lms.push_back(lmptr);
						lmtrack_final.scale_change.push_back(0);
						++lmtrack_final.n_pts;

						this->saveLandmark(lmptr);
					}
				}
			}

			// lms1_final 중, depth가 복원되지 않은 경우 복원해준다.
			uint32_t cnt_recon = 0 ;
			for(const auto& lm : lmtrack_final.lms)
			{
				if( !lm->isTriangulated() && lm->getLastParallax() >= THRES_PARALLAX)
				{
					if( lm->getObservations().size() != lm->getRelatedFramePtr().size() )
						throw std::runtime_error("lm->getObservations().size() != lm->getRelatedFramePtr().size()\n");

					const Pixel&   pt0 = lm->getObservations().front(), pt1 = lm->getObservations().back();
					const PoseSE3& Tw0 = lm->getRelatedFramePtr().front()->getPose(), T1w = lm->getRelatedFramePtr().back()->getPoseInv();

					PoseSE3 T10 = T1w * Tw0;
					const Rot3& R10 = T10.block<3,3>(0,0);
					const Pos3& t10 = T10.block<3,1>(0,3);

					// Reconstruct points
					Point X0, X1;
					mapping::triangulateDLT(pt0, pt1, R10, t10, cam_, X0, X1);

					if(X0(2) > 0) {
						Point Xworld = Tw0.block<3,3>(0,0) * X0 + Tw0.block<3,1>(0,3);
						lm->set3DPoint(Xworld);
						++cnt_recon;
					}
				}
			}
			std::cout << " Recon done. : " << cnt_recon << "\n";

			// lms1와 pts1을 frame_curr에 넣는다.
			frame_curr->setPtsSeenAndRelatedLandmarks(lmtrack_final.pts1, lmtrack_final.lms);


			std::cout <<" VISUAL ODOMETRY IS INITIALIZED..!\n";
			
			system_flags_.flagVOInit = true;
		}
	}
	else 
	{
		// 초기화 완료.
		/* 
				========================================================================
				========================================================================
				========================================================================
				========================================================================
				========================================================================
				========================================================================
				========================================================================


											알고리즘 계속 구동.


				========================================================================
				========================================================================
				========================================================================
				========================================================================
				========================================================================
				========================================================================
				========================================================================
		*/

		timer::tic();

		// VO initialized. Do track the new image. (only get 'alive()' landmarks)
		LandmarkTracking lmtrack_prev(frame_prev_->getPtsSeen(), frame_prev_->getPtsSeen(), frame_prev_->getRelatedLandmarkPtr());

		// 이전 자세의 변화량을 가져온다. 
		PoseSE3 Twc_prev   = frame_prev_->getPose();
		PoseSE3 Tcw_prev   = frame_prev_->getPoseInv();

		PoseSE3 dT01_prior = frame_prev_->getPoseDiff01();
		PoseSE3 Twc_prior  = Twc_prev * dT01_prior;
		PoseSE3 Tcw_prior  = geometry::inverseSE3_f(Twc_prior);
		
		std::cout << " CURRENT REFERENCE KEYFRAME : " << keyframe_ref_->getID() << std::endl;

		// Make tracking prior & estimated scale
		for(int i = 0; i < lmtrack_prev.n_pts; ++i)
		{
			const LandmarkPtr& lm = lmtrack_prev.lms[i];

			float patch_scale = 1.0f;
			if( lm->isBundled() )
			{
				const Point& Xw = lm->get3DPoint();
				Point Xp = Tcw_prev.block<3,3>(0,0)*Xw + Tcw_prev.block<3,1>(0,3);
				Point Xc = Tcw_prior.block<3,3>(0,0)*Xw + Tcw_prior.block<3,1>(0,3);
		
				patch_scale = Xp(2)/Xc(2);
				
				if(Xc(2) > 0) 
					lmtrack_prev.pts1[i] = cam_->projectToPixel(Xc);
				else 
					lmtrack_prev.pts1[i] = lmtrack_prev.pts0[i];
			}
			else 
				lmtrack_prev.pts1[i] = lmtrack_prev.pts0[i];

			lmtrack_prev.scale_change[i] = patch_scale;
		}
		std::cout << colorcode::text_green << "Time [track preliminary]: " << timer::toc(0) << " [ms]\n" << colorcode::cout_reset;
		
		// frame_prev_ 의 lms 를 현재 이미지로 track. 5ms
		timer::tic();
		MaskVec  mask_track;
		tracker_->trackBidirectionWithPrior(I0, I1, lmtrack_prev.pts0, params_.feature_tracker.window_size, params_.feature_tracker.max_level, params_.feature_tracker.thres_error, params_.feature_tracker.thres_bidirection,
			lmtrack_prev.pts1, mask_track);
		std::cout << colorcode::text_green << "Time [track bidirection]: " << timer::toc(0) << " [ms]\n" << colorcode::cout_reset;

		LandmarkTracking lmtrack_kltok(lmtrack_prev, mask_track); // make valid landmark

		// Scale refinement 50ms
		timer::tic();
		MaskVec mask_refine(lmtrack_kltok.pts0.size(),true);
		const cv::Mat& du0 = frame_prev_->getImageDu();
		const cv::Mat& dv0 = frame_prev_->getImageDv();
		tracker_->trackWithScale(
			I0, du0, dv0, I1, 
			lmtrack_kltok.pts0, lmtrack_kltok.scale_change, lmtrack_kltok.pts1,
			mask_refine); // TODO (SCALE + Position KLT)

		LandmarkTracking lmtrack_scaleok(lmtrack_kltok, mask_refine);
		std::cout << colorcode::text_green << "Time [trackWithScale   ]: " << timer::toc(0) << " [ms]\n" << colorcode::cout_reset;
		

		// Motion estimation
		// 깊이를 가진 점 갯수를 세어보고, 20개 이상이면 local bundle을 수행한다.
		timer::tic();

		const Rot3& Rcw_prev = Tcw_prev.block<3,3>(0,0);
		const Pos3& tcw_prev = Tcw_prev.block<3,1>(0,3);

		std::vector<int> index_ba;
		if(keyframes_->getList().size() > 5) {
			// # of keyframes is over 5 (키프레임이 많으면, bundled point만 사용한다.)
			for(int i = 0; i < lmtrack_scaleok.n_pts; ++i) {
				const LandmarkPtr& lm = lmtrack_scaleok.lms[i];
				if( lm->isBundled() ) {
					Point Xp = Rcw_prev * lm->get3DPoint() + tcw_prev;
					if(Xp(2) > 0.1) index_ba.push_back(i);
				}
			}
		}
		else {
			for(int i = 0; i < lmtrack_scaleok.n_pts; ++i) {
				const LandmarkPtr& lm = lmtrack_scaleok.lms[i];
				if( lm->isTriangulated() ) {
					Point Xp = Rcw_prev * lm->get3DPoint() + tcw_prev;
					if(Xp(2) > 0.1) index_ba.push_back(i);
				}
			}
		}
		
		MaskVec mask_motion(lmtrack_scaleok.n_pts, true); // pose-only BA로 가면 size가 줄어든다...
		Rot3 dR10; Pos3 dt10; PoseSE3 dT10;
		Rot3 dR01; Pos3 dt01; PoseSE3 dT01;
		bool poseonlyBA_success = false;
		bool has_sufficient_points = index_ba.size() > 10;
		if(has_sufficient_points)
		{
			// Do Local BA
			int n_pts_ba = index_ba.size();
			std::cout << colorcode::text_magenta << "DO pose-only Bundle Adjustment... with [" << n_pts_ba <<"] points.\n" << colorcode::cout_reset;

			PixelVec pts1_ba(n_pts_ba);
			PixelVec pts1_proj_ba(n_pts_ba);
			PointVec Xp_ba(n_pts_ba);
			MaskVec  mask_ba(n_pts_ba, true);
			for(int i = 0; i < n_pts_ba; ++i)
			{
				const int& idx_tmp = index_ba[i];
				const LandmarkPtr& lm = lmtrack_scaleok.lms[idx_tmp];
				const Pixel&      pt1 = lmtrack_scaleok.pts1[idx_tmp];
				const Point&        X = lm->get3DPoint();

				Point Xp = Rcw_prev * X + tcw_prev;

				pts1_ba[i] = pt1;
				Xp_ba[i]   = Xp;
			}

			// Do pose-only BA for current frame.
			dR01 = dT01_prior.block<3,3>(0,0); 
			dt01 = dT01_prior.block<3,1>(0,3);
			poseonlyBA_success = motion_estimator_->poseOnlyBundleAdjustment(
								Xp_ba, pts1_ba, cam_, params_.motion_estimator.thres_poseba_error,
								dR01, dt01, mask_ba);

			if( poseonlyBA_success )
			{	
				// Set mask
				for(int i = 0; i < index_ba.size(); ++i){
					const int& idx = index_ba[i];
					if( mask_ba[i] ) mask_motion[idx] = true;
					else mask_motion[idx] = false;
				}

				dT01 << dR01, dt01, 0,0,0,1;
				dT10 = geometry::inverseSE3_f(dT01);

				dR10 = dT10.block<3,3>(0,0);
				dt10 = dT10.block<3,1>(0,3);

				if(std::isnan(dt10.norm()))
					throw std::runtime_error("std::isnan(dt01.norm()) ...");
				
				frame_curr->setPose(Twc_prev*dT01);		
				frame_curr->setPoseDiff10(dT10);	

				// Projection 
				for(int i = 0; i < Xp_ba.size(); ++i)
				{
					Point Xc = dR10*Xp_ba[i] + dt10;
					pts1_proj_ba[i] = cam_->projectToPixel(Xc);
				}

				std::cout <<"     === prior --> est dt01: " 
					<< dT01_prior.block<3,1>(0,3).transpose() << " --> "
					<< dt01.transpose() <<std::endl;
				
				if( true )
					this->showTrackingBA("img_feautues", I1, pts1_ba, pts1_proj_ba); // show motion estimation result
			}
		}

		// If pose-only BA is failed, do 5-point algorithm.
		if( !poseonlyBA_success ) 
		{ 
			frame_curr->setPoseOnlyFailed();

			// do 5 point algorihtm (scale is of the previous frame)
			if(!has_sufficient_points)
			{
				std::cout << colorcode::text_red;
				std::cout << "\n\n\n";
				std::cout << "!!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!!\n";
				std::cout << " !!! !!! !! WARNING ! Because of insufficient points, 5-points algorithm runs... !! !!! !!! \n";
				std::cout << "!!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!!\n\n\n";
				std::cout << colorcode::cout_reset << std::endl;
			}
			else
			{
				std::cout << colorcode::text_red;
				std::cout << "\n\n\n";
				std::cout << "!!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!!\n";
				std::cout << " !!! !!! !! WARNING ! Because of pose-only BA is failed, 5-points algorithm runs... !! !!! !!! \n";
				std::cout << "!!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!!\n\n\n";
				std::cout << colorcode::cout_reset << std::endl;
			}
			

			PointVec X0_inlier(lmtrack_scaleok.n_pts);

			bool fivepoints_success = false;
			fivepoints_success = motion_estimator_->calcPose5PointsAlgorithm(
				lmtrack_scaleok.pts0, lmtrack_scaleok.pts1, cam_, dR10, dt10, X0_inlier, mask_motion);
			
			if( !fivepoints_success ) 
				throw std::runtime_error("'calcPose5PointsAlgorithm()' is failed. Terminate the algorithm.");

			// Frame_curr의 자세를 넣는다.
			float scale = frame_prev_->getPoseDiff01().block<3,1>(0,3).norm();
			dT10 << dR10, (scale/dt10.norm())*dt10, 0.0f, 0.0f, 0.0f, 1.0f;
			dT01 = geometry::inverseSE3_f(dT10);

			frame_curr->setPose(Twc_prev*dT01);		
			frame_curr->setPoseDiff10(dT10);	
		}

		LandmarkTracking lmtrack_motion(lmtrack_scaleok, mask_motion);
		std::cout << colorcode::text_green << "Time [track Motion Est.]: " << timer::toc(0) << " [ms]\n" << colorcode::cout_reset;
		
		// Check sampson distance 0.01 ms
		std::vector<float> symm_epi_dist;
		motion_estimator_->calcSampsonDistance(lmtrack_motion.pts0, lmtrack_motion.pts1, cam_, 
												dT10.block<3,3>(0,0), dT10.block<3,1>(0,3), symm_epi_dist);

		MaskVec mask_sampson(lmtrack_motion.n_pts, true);
		for(int i = 0; i < mask_sampson.size(); ++i)
			mask_sampson[i] = symm_epi_dist[i] < params_.feature_tracker.thres_sampson;
		

		// Done. Add observations.
		LandmarkTracking lmtrack_final(lmtrack_motion, mask_sampson);
		for(int i = 0; i < lmtrack_final.pts1.size(); ++i)
			lmtrack_final.lms[i]->addObservationAndRelatedFrame(lmtrack_final.pts1[i], frame_curr);
				
	
#ifdef RECORD_FRAME_STAT
statcurr_frame.Twc   = frame_curr->getPose();
statcurr_frame.Tcw   = frame_curr->getPoseInv();
statcurr_frame.dT_10 = frame_curr->getPoseDiff10();
statcurr_frame.dT_01 = frame_curr->getPoseDiff01();
#endif
			
		// 빈 곳에 특징점 pts1_new 를 추출한다. 2 ms
		timer::tic();
		PixelVec pts1_new;
		extractor_->updateWeightBin(lmtrack_final.pts1); // 이미 pts1가 있는 곳은 제외.
		extractor_->extractORBwithBinning_fast(I1, pts1_new, true);
		std::cout << colorcode::text_green << "Time [extract ORB      ]: " << timer::toc(0) << " [ms]\n" << colorcode::cout_reset;
		
		timer::tic();
		if( pts1_new.size() > 0 ){
			// 새로운 특징점을 back-track.
			PixelVec pts0_new;
			MaskVec  mask_new;
			tracker_->trackBidirection(I1, I0, pts1_new, params_.feature_tracker.window_size, params_.feature_tracker.max_level, params_.feature_tracker.thres_error, params_.feature_tracker.thres_bidirection,
				pts0_new, mask_new);

			// 새로운 특징점은 새로운 landmark가 된다.
			for(int i = 0; i < pts1_new.size(); ++i) 
			{
				if( mask_new[i] )
				{
					const Pixel& p0_new = pts0_new[i];
					const Pixel& p1_new = pts1_new[i];

					LandmarkPtr lmptr = std::make_shared<Landmark>(p0_new, frame_prev_, cam_);
					lmptr->addObservationAndRelatedFrame(p1_new, frame_curr);

					lmtrack_final.pts0.push_back(p0_new);
					lmtrack_final.pts1.push_back(p1_new);
					lmtrack_final.lms.push_back(lmptr);
					lmtrack_final.scale_change.push_back(0);
					++lmtrack_final.n_pts;

					this->saveLandmark(lmptr);
				}
			}
		}
		std::cout << colorcode::text_green << "Time [New fts ext track]: " << timer::toc(0) << " [ms]\n" << colorcode::cout_reset;
		
		
		// lms1와 pts1을 frame_curr에 넣는다.
		frame_curr->setPtsSeenAndRelatedLandmarks(lmtrack_final.pts1, lmtrack_final.lms);
	}

	// Check keyframe update rules.
	bool flag_add_new_keyframe = this->keyframes_->checkUpdateRule(frame_curr);

	if( flag_add_new_keyframe ) // 새로운 키프레임 추가.
	{
		// Add new keyframe
		timer::tic();
		this->saveKeyframe(frame_curr);
		this->keyframes_->addNewKeyframe(frame_curr);
		this->keyframe_ref_ = frame_curr;

		// Reconstruct map points. 새로 만들어진 keyframe에서 보인 landmarks 중 reconstruction이 되지 않은 경우, DLT로 초기화 해준다.
		uint32_t cnt_recon = 0;
		for(const auto& lm : frame_curr->getRelatedLandmarkPtr())
		{
			if( lm->isAlive()
			&& !lm->isTriangulated() 
			&& lm->getLastParallax() >= THRES_PARALLAX )
			{
				if( lm->getObservationsOnKeyframes().size() > 2 )
				{
					// 3번 이상 keyframe에서 보였다.
					const Pixel& pt0 = lm->getObservationsOnKeyframes().front();
					const Pixel& pt1 = lm->getObservationsOnKeyframes().back();

					const PoseSE3& Tw0 = lm->getRelatedKeyframePtr().front()->getPose();
					const PoseSE3& T1w = lm->getRelatedKeyframePtr().back()->getPoseInv();
					PoseSE3 T10_tmp = T1w * Tw0;
					const Rot3& R10_tmp = T10_tmp.block<3,3>(0,0);
					const Pos3& t10_tmp = T10_tmp.block<3,1>(0,3);

					// Reconstruct points
					Point X0, X1;
					mapping::triangulateDLT(pt0, pt1, R10_tmp, t10_tmp, cam_, X0, X1);

					// Check reprojection error for the first image
					Pixel pt0_proj = cam_->projectToPixel(X0);
					Pixel dpt0 = pt0 - pt0_proj;
					float dpt0_norm2 = dpt0.x*dpt0.x + dpt0.y*dpt0.y;
					if(dpt0_norm2 > 1.0) continue;

					Pixel pt1_proj = cam_->projectToPixel(X1);
					Pixel dpt1 = pt1 - pt1_proj;
					float dpt1_norm2 = dpt1.x*dpt1.x + dpt1.y*dpt1.y;
					if(dpt1_norm2 > 1.0) continue;

					// Check the point in front of cameras
					if(X0(2) > 0 && X1(2) > 0) 
					{
						Point Xworld = Tw0.block<3,3>(0,0)*X0 + Tw0.block<3,1>(0,3);
						lm->set3DPoint(Xworld);
						++cnt_recon;
					}
				}
			}
		}
		std::cout << "    # of newly reconstructed points: " << cnt_recon << std::endl;

		// Local Bundle Adjustment
		motion_estimator_->localBundleAdjustmentSparseSolver(keyframes_, cam_);
		std::cout << colorcode::text_green << "Time [keyframe addition]: " << timer::toc(0) << " [ms]\n" << colorcode::cout_reset;

		// Make variables for refine tracking
		const LandmarkPtrVec& lms_final = frame_curr->getRelatedLandmarkPtr();
		FloatVec scale_estimated(lms_final.size(), 1.0f);
		MaskVec mask_final(lms_final.size(), true);
		PixelVec pts_refine(lms_final.size());

		for(int i = 0; i < lms_final.size(); ++i){
			// Estimate current image-patch-scale...
			const LandmarkPtr& lm = lms_final[i];
			if(lm->isTriangulated()) {
				const Point& Xw = lm->get3DPoint();

				const PoseSE3& T0w = lm->getRelatedFramePtr().front()->getPoseInv();
				const PoseSE3& T1w = lm->getRelatedFramePtr().back()->getPoseInv();

				Point X0  = T0w.block<3,3>(0,0)*Xw + T0w.block<3,1>(0,3);
				Point X1  = T1w.block<3,3>(0,0)*Xw + T1w.block<3,1>(0,3);

				float d0 = X0(2), d1 = X1(2);
				float scale = d0/d1;

				mask_final[i]      = true;
				scale_estimated[i] = scale;
				pts_refine[i]      = lm->getObservations().back();

				// std::cout << i << "-th point:" << Xw.transpose() << " scale: " << scale << std::endl;
			}
			else 
				mask_final[i] = false;
		}

		// Refine the tracking results
		timer::tic();
		tracker_->refineTrackWithScale(I1, lms_final, scale_estimated, pts_refine, mask_final);
		std::cout << colorcode::text_green << "Time [trackWithScale   ]: " << timer::toc(0) << " [ms]\n" << colorcode::cout_reset;

		// Update points
		for(int i = 0; i < lms_final.size(); ++i)
		{
			const LandmarkPtr& lm = lms_final[i];
			if(mask_final[i])
				lm->changeLastObservation(pts_refine[i]);
		}

#ifdef RECORD_KEYFRAME_STAT
timer::tic();
PointVec X_tmp;
const LandmarkPtrVec& lmvec_tmp = frame_curr->getRelatedLandmarkPtr();
for(int i = 0; i < lmvec_tmp.size(); ++i)
{
	X_tmp.push_back(lmvec_tmp[i]->get3DPoint());
}
statcurr_keyframe.Twc = frame_curr->getPose();
statcurr_keyframe.mappoints = X_tmp;
stat_.stats_keyframe.push_back(statcurr_keyframe);

for(int j = 0; j < stat_.stats_keyframe.size(); ++j){
	stat_.stats_keyframe[j].Twc = all_keyframes_[j]->getPose();

	const LandmarkPtrVec& lmvec_tmp = all_keyframes_[j]->getRelatedLandmarkPtr();
	stat_.stats_keyframe[j].mappoints.resize(lmvec_tmp.size());
	for(int i = 0; i < lmvec_tmp.size(); ++i) {
		stat_.stats_keyframe[j].mappoints[i] = lmvec_tmp[i]->get3DPoint();
	}
}
std::cout << colorcode::text_green << "Time [RECORD KEYFR STAT]: " << timer::toc(0) << " [ms]\n" << colorcode::cout_reset;
#endif





	} // KEYFRAME addition done.



	
	// Replace the 'frame_prev_' with 'frame_curr'
	this->frame_prev_ = frame_curr;
	if(this->keyframe_ref_ == nullptr) this->keyframe_ref_ = this->frame_prev_;

	// Visualization 3D points
	PointVec X_world_recon;
	X_world_recon.reserve(all_landmarks_.size());
	for(const auto& lm : all_landmarks_)
		if(lm->isTriangulated()) 
			X_world_recon.push_back(lm->get3DPoint());
	
	std::cout << "# of all landmarks: " << X_world_recon.size() << std::endl;

#ifdef RECORD_FRAME_STAT
statcurr_frame.mappoints.resize(0);
statcurr_frame.mappoints = X_world_recon;
#endif

	// Update statistics
	stat_.stats_landmark.push_back(statcurr_landmark);
	// stat_.stats_frame.resize(0);
	stat_.stats_frame.push_back(statcurr_frame);
	
	for(int j = 0; j < this->all_frames_.size(); ++j)
		stat_.stats_frame[j].Twc = all_frames_[j]->getPose();

	stat_.stats_execution.push_back(statcurr_execution);
	std::cout << "Statistics Updated. size: " << stat_.stats_landmark.size() << "\n";

	// Notify a thread.
	// mut_scale_estimator_->lock();
	// mut_scale_estimator_->unlock();
	// cond_var_scale_estimator_->notify_all();
};