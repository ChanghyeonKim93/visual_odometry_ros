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
: cam_(nullptr), system_flags_(), dataset_(), frame_prev_(nullptr) 
{
	std::cout << "Scale mono VO starts\n";
		
	// Initialize camera
	cam_ = std::make_shared<Camera>();
	Landmark::cam_       = cam_;
	Frame::cam_          = cam_;
	ScaleEstimator::cam_ = cam_;

	if(mode == "dataset"){
		std::cout <<"ScaleMonoVO - 'dataset' mode.\n";
		std::string dir_dataset = "D:/#DATASET/kitti/data_odometry_gray";
		std::string dataset_num = "00";

		this->loadCameraIntrinsicAndUserParameters_KITTI_IMAGE0(dir_dataset + "/dataset/sequences/" + dataset_num + "/intrinsic.yaml");

		// Get dataset filenames.
		dataset_loader::getImageFileNames_KITTI(dir_dataset, dataset_num, dataset_);

		runDataset(); // run while loop
	}
	else if(mode == "rosbag"){
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
	mut_scale_estimator_      = std::make_shared<std::mutex>();
	cond_var_scale_estimator_ = std::make_shared<std::condition_variable>(); // New pose 가 도
	flag_do_ASR_              = std::make_shared<bool>(false);
	scale_estimator_          = std::make_shared<ScaleEstimator>(mut_scale_estimator_, cond_var_scale_estimator_, flag_do_ASR_);
	
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
 * @brief Run scale mono vo on the dataset.
 * @details 낱장 이미지 파일들로 저장되어있는 (ex. KITTI) 데이터에 대해 scale mono vo 알고리즘 구동. 
 * @return void
 * @author Changhyeon Kim (hyun91015@gmail.com)
 * @date 10-July-2022
 */
void ScaleMonoVO::runDataset() {
	
	cv::Mat img0 = cv::imread(dataset_.image_names[0], cv::IMREAD_GRAYSCALE);
	cv::Mat img0f;
	img0.convertTo(img0f, CV_32FC1);
	std::cout << "input image type: " << image_processing::type2str(img0f) << std::endl;
	/*while (true) {

	}*/
};

/**
 * @brief load monocular camera intrinsic parameters from yaml file.
 * @details 카메라의 intrinsic parameter (fx,fy,cx,cy, distortion)을 얻어온다. 
 * @param dir file directory
 * @return void
 * @author Changhyeon Kim (hyun91015@gmail.com)
 * @date 21-July-2022
 */
void ScaleMonoVO::loadCameraIntrinsicAndUserParameters_KITTI_IMAGE0(const std::string& dir) {

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
	params_.feature_tracker.thres_error = fs["feature_tracker.thres_error"];
	params_.feature_tracker.thres_bidirection = fs["feature_tracker.thres_bidirection"];
	params_.feature_tracker.window_size = (int)fs["feature_tracker.window_size"];
	params_.feature_tracker.max_level = (int)fs["feature_tracker.max_level"];


	params_.feature_extractor.n_features = (int)fs["feature_extractor.n_features"];
	params_.feature_extractor.n_bins_u   = (int)fs["feature_extractor.n_bins_u"];
	params_.feature_extractor.n_bins_v   = (int)fs["feature_extractor.n_bins_v"];
	params_.feature_extractor.thres_fastscore = fs["feature_extractor.thres_fastscore"];
	params_.feature_extractor.radius          = fs["feature_extractor.radius"];

	params_.keyframe_update.thres_alive_ratio   = fs["keyframe_update.thres_alive_ratio"];
	params_.keyframe_update.thres_mean_parallax = fs["keyframe_update.thres_mean_parallax"];
	
	params_.map_update.thres_parallax = fs["map_update.thres_parallax"];
	params_.map_update.thres_parallax *= D2R;

	std::cout << " - 'loadCameraIntrinsicMono()' - loaded.\n";
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
 * @brief function to track a new image
 * @details 새로 들어온 이미지의 자세를 구하는 함수. 만약, scale mono vo가 초기화되지 않은 경우, 해당 이미지를 초기 이미지로 설정. 
 * @param img 입력 이미지 (CV_8UC1)
 * @param timestamp 입력 이미지의 timestamp.
 * @return none
 * @author Changhyeon Kim (hyun91015@gmail.com)
 * @date 10-July-2022
 */
void ScaleMonoVO::trackImage(const cv::Mat& img, const double& timestamp){
	// Generate statistics
	AlgorithmStatistics::LandmarkStatistics  statcurr_landmark;
	AlgorithmStatistics::FrameStatistics     statcurr_frame;
	AlgorithmStatistics::ExecutionStatistics statcurr_execution;

			
	// 현재 이미지에 대한 새로운 Frame 생성
	FramePtr frame_curr = std::make_shared<Frame>();
	this->saveFrames(frame_curr);
	
	// 이미지 undistort (KITTI라서 할 필요 X)
	cv::Mat img_undist;
	if(system_flags_.flagDoUndistortion) {
		cam_->undistortImage(img, img_undist);
		img_undist.convertTo(img_undist, CV_8UC1);
	}
	else img.copyTo(img_undist);

	// frame_curr에 img_undist와 시간 부여
	frame_curr->setImageAndTimestamp(img_undist, timestamp);

	if( !system_flags_.flagVOInit ) { // 초기화 미완료
		if( !system_flags_.flagFirstImageGot ) { // 최초 이미지	
			// Get the first image
			const cv::Mat& I0 = frame_curr->getImage();

			// Extract pixels
			PixelVec       pxvec0;
			LandmarkPtrVec lmvec0;

			extractor_->resetWeightBin();
			extractor_->extractORBwithBinning(I0, pxvec0, true);
#ifdef RECORD_EXECUTION_STAT
	statcurr_execution.time_new = 0;
#endif

#ifdef RECORD_LANDMARK_STAT
	// get statistics
	uint32_t n_pts = pxvec0.size();
	statcurr_landmark.n_initial = n_pts;
	statcurr_landmark.n_pass_bidirection = n_pts;
	statcurr_landmark.n_pass_1p = n_pts;
	statcurr_landmark.n_pass_5p = n_pts;
	statcurr_landmark.n_new = n_pts;
	statcurr_landmark.n_final = n_pts;
	
	// statcurr_landmark.max_age = 1;
	// statcurr_landmark.min_age = 1;
	statcurr_landmark.avg_age = 1.0f;

	statcurr_landmark.n_ok_parallax = 0;
	// statcurr_landmark.min_parallax  = 0.0;
	// statcurr_landmark.max_parallax  = 0.0;
	statcurr_landmark.avg_parallax  = 0.0;
#endif

#ifdef RECORD_EXECUTION_STAT
	statcurr_execution.time_track = statcurr_execution.time_new;
	statcurr_execution.time_1p = statcurr_execution.time_new;
	statcurr_execution.time_5p = statcurr_execution.time_new;
#endif
			// 초기 landmark 생성
			lmvec0.reserve(pxvec0.size());
			for(auto p : pxvec0) 
				lmvec0.push_back(std::make_shared<Landmark>(p, frame_curr));
			
			// Related Landmark와 tracked pixels를 업데이트
			frame_curr->setPtsSeenAndRelatedLandmarks(pxvec0, lmvec0);
			
			frame_curr->setPose(PoseSE3::Identity());
			frame_curr->setPoseDiff10(PoseSE3::Identity());
			
			this->saveLandmarks(lmvec0);	

			if( true ){
				this->showTracking("img_features", frame_curr->getImage(), pxvec0, PixelVec(), PixelVec());
			}
			// 첫 이미지 업데이트 완료
			system_flags_.flagFirstImageGot = true;
		}
		else { 
			// 최초 첫 이미지는 들어왔으나, 아직 초기화가 되지 않은 상태.
			// 초기화는 맨 첫 이미지 (첫 키프레임) 대비, 제대로 추적 된 landmark가 60 퍼센트 이상이며, 
			// 추적된 landmark 각각의 최대 parallax 가 1도 이상인 경우 초기화 완료.	

			// 이전 프레임의 pixels 와 lmvec0을 가져온다.
			const PixelVec&       pxvec0 = frame_prev_->getPtsSeen();
			const LandmarkPtrVec& lmvec0 = frame_prev_->getRelatedLandmarkPtr();
			const cv::Mat&        I0     = frame_prev_->getImage();
			const cv::Mat&        I1     = frame_curr->getImage();

#ifdef RECORD_LANDMARK_STAT
	statcurr_landmark.n_initial = pxvec0.size();
#endif

#ifdef RECORD_EXECUTION_STAT
	timer::tic();
#endif
			// frame_prev_ 의 lms 를 현재 이미지로 track.
			PixelVec pxvec1_track;
			MaskVec  maskvec1_track;
			tracker_->trackBidirection(I0, I1, pxvec0, params_.feature_tracker.window_size, params_.feature_tracker.max_level, params_.feature_tracker.thres_error, params_.feature_tracker.thres_bidirection,
							           pxvec1_track, maskvec1_track);
#ifdef RECORD_EXECUTION_STAT
	statcurr_execution.time_track = timer::toc(false);
#endif

			// Tracking 결과를 반영하여 pxvec1_alive, lmvec1_alive를 정리한다.
			PixelVec       pxvec0_alive;
			PixelVec       pxvec1_alive;
			LandmarkPtrVec lmvec1_alive;
			int cnt_alive = 0;
			for(int i = 0; i < pxvec1_track.size(); ++i){
				if( maskvec1_track[i]) {
					pxvec0_alive.push_back(pxvec0[i]);
					pxvec1_alive.push_back(pxvec1_track[i]);
					lmvec1_alive.push_back(lmvec0[i]);
					++cnt_alive;
				}
				else lmvec0[i]->setDead(); // track failed. Dead point.
			}
		
#ifdef RECORD_LANDMARK_STAT
statcurr_landmark.n_pass_bidirection = cnt_alive;
#endif


#ifdef RECORD_EXECUTION_STAT 
timer::tic(); 
#endif
			// 1-point RANSAC 을 이용하여 outlier를 제거 & tentative steering angle 구함.
			MaskVec maskvec_1p;
			float steering_angle_curr = motion_estimator_->findInliers1PointHistogram(pxvec0_alive, pxvec1_alive, cam_, maskvec_1p);
			

#ifdef RECORD_EXECUTION_STAT
statcurr_execution.time_1p = timer::toc(false);
#endif
			PixelVec       pxvec0_1p;
			PixelVec       pxvec1_1p;
			LandmarkPtrVec lmvec1_1p;
			int cnt_1p = 0;
			for(int i = 0; i < maskvec_1p.size(); ++i){
				if( maskvec_1p[i]) {
					pxvec0_1p.push_back(pxvec0_alive[i]);
					pxvec1_1p.push_back(pxvec1_alive[i]);
					lmvec1_1p.push_back(lmvec1_alive[i]);
					++cnt_1p;
				}
				else lmvec1_alive[i]->setDead(); // track failed. Dead point.
			}


#ifdef RECORD_LANDMARK_STAT
statcurr_landmark.n_pass_1p = cnt_1p;
#endif


#ifdef RECORD_EXECUTION_STAT
timer::tic();
#endif
			// pts0 와 pts1을 이용, 5-point algorithm 으로 모션 & X0 를 구한다.
			// 만약 mean optical flow의 중간값이 약 1 px 이하인 경우, 정지 상태로 가정하고 스킵.
			MaskVec maskvec_inlier(pxvec0_1p.size());
			PointVec X0_inlier(pxvec0_1p.size());
			Rot3 dR10;
			Pos3 dt10;
			if( !motion_estimator_->calcPose5PointsAlgorithm(pxvec0_1p, pxvec1_1p, cam_, dR10, dt10, X0_inlier, maskvec_inlier) ) {
				throw std::runtime_error("calcPose5PointsAlgorithm() is failed.");
			}
#ifdef RECORD_EXECUTION_STAT
statcurr_execution.time_5p = timer::toc(false);
#endif			
			// Frame_curr의 자세를 넣는다.
			float scale;
			if(frame_curr->getID() > 300) scale = 0.22;
			else scale = 0.9;
			PoseSE3 dT10; dT10 << dR10, scale*dt10, 0.0f, 0.0f, 0.0f, 1.0f;
			PoseSE3 dT01 = dT10.inverse();

			frame_curr->setPose(frame_prev_->getPose()*dT01);		
			frame_curr->setPoseDiff10(dT10);		

			// Steering angle을 계산한다.
			steering_angle_curr = scale_estimator_->calcSteeringAngleFromRotationMat(dR10.transpose());
			frame_curr->setSteeringAngle(steering_angle_curr);

			// Detect turn region by a steering angle.
			if(scale_estimator_->detectTurnRegions(frame_curr)){
				FramePtrVec frames_turn_tmp;
				frames_turn_tmp = scale_estimator_->getAllTurnRegions();
				for(auto f :frames_turn_tmp)
					stat_.stat_turn.turn_regions.push_back(f);
			}

#ifdef RECORD_FRAME_STAT
statcurr_frame.steering_angle = steering_angle_curr;
#endif

			// tracking, 5p algorithm, newpoint 모두 합쳐서 살아남은 점만 frame_curr에 넣는다
			float avg_flow = 0.0f;
			PixelVec       pxvec0_final;
			PixelVec       pxvec1_final;
			LandmarkPtrVec lmvec1_final;
			cnt_alive = 0;
			int cnt_parallax_ok = 0;
			for(int i = 0; i < pxvec0_1p.size(); ++i){
				if( maskvec_inlier[i] ) {
					lmvec1_1p[i]->addObservationAndRelatedFrame(pxvec1_1p[i], frame_curr);
					avg_flow += lmvec1_1p[i]->getLastOptFlow();
					if(lmvec1_1p[i]->getMaxParallax() > params_.map_update.thres_parallax) {
						++cnt_parallax_ok;
						// lmvec1_1p[i]->set3DPoint(X0_inlier[i]);
					}
					pxvec0_final.push_back(pxvec0_1p[i]);
					pxvec1_final.push_back(pxvec1_1p[i]);
					lmvec1_final.push_back(lmvec1_1p[i]);
					++cnt_alive;
				}
				else lmvec1_1p[i]->setDead(); // 5p algorithm failed. Dead point.
			}
			avg_flow /= (float) cnt_alive;
			std::cout << " AVERAGE FLOW : " << avg_flow << " px\n";
			std::cout << " Parallax OK : " << cnt_parallax_ok << std::endl;
			// Scale forward propagation
			if(frame_curr->getID() > 3 && avg_flow > 1.5)
				scale_estimator_->module_ScaleForwardPropagation(lmvec1_final, all_frames_,dT10);

#ifdef RECORD_FRAME_STAT
statcurr_frame.Twc = frame_curr->getPose();
statcurr_frame.Tcw = frame_curr->getPose().inverse();
statcurr_frame.dT_10 = frame_curr->getPoseDiff10();
statcurr_frame.dT_01 = frame_curr->getPoseDiff01();
#endif

#ifdef RECORD_LANDMARK_STAT
statcurr_landmark.n_pass_5p = cnt_alive;
#endif

			// lmvec1_final 중, depth가 복원되지 않은 경우 복원해준다.
#ifdef RECORD_LANDMARK_STAT
statcurr_landmark.n_ok_parallax = cnt_parallax_ok;
#endif

#ifdef RECORD_EXECUTION_STAT
timer::tic();
#endif
			// 빈 곳에 특징점 pts1_new 를 추출한다.
			PixelVec pxvec1_new;
			extractor_->updateWeightBin(pxvec1_final); // 이미 pts1가 있는 곳은 제외.
			extractor_->extractORBwithBinning(frame_curr->getImage(), pxvec1_new, true);
#ifdef RECORD_EXECUTION_STAT
statcurr_execution.time_new = timer::toc(false);
statcurr_execution.time_total = statcurr_execution.time_new + statcurr_execution.time_track + statcurr_execution.time_1p + statcurr_execution.time_5p;
#endif		

#ifdef RECORD_LANDMARK_STAT
statcurr_landmark.n_new = pxvec1_new.size();
#endif
			if( true ){
				// this->showTracking("img_features", frame_curr->getImage(), pxvec0_final, pxvec1_final, pxvec1_new);
				this->showTracking("img_features", frame_curr->getImage(), lmvec1_final);
			}

			if( pxvec1_new.size() > 0 ){
				// 새로운 특징점은 새로운 landmark가 된다.
				for(auto p1_new : pxvec1_new) {
					LandmarkPtr ptr = std::make_shared<Landmark>(p1_new, frame_curr);
					pxvec1_final.emplace_back(p1_new);
					lmvec1_final.push_back(ptr);
					this->saveLandmarks(ptr);	
				}
			}

			// lms1와 pts1을 frame_curr에 넣는다.
			frame_curr->setPtsSeenAndRelatedLandmarks(pxvec1_final, lmvec1_final);

#ifdef RECORD_LANDMARK_STAT
	statcurr_landmark.n_final = pxvec1_final.size();
#endif

			float avg_age = calcLandmarksMeanAge(lmvec1_final);
#ifdef RECORD_LANDMARK_STAT
	statcurr_landmark.avg_age = avg_age;
#endif
			// 초기화를 완료할지 판단
			// lmvec1_final가 최초 관측되었던 (keyframe) 
			bool initialization_done = false;
			int n_lms_alive       = 0;
			int n_lms_parallax_ok = 0;
			float mean_parallax   = 0;
			for(int i = 0; i < lmvec1_final.size(); ++i){
				const LandmarkPtr& lm = lmvec1_final[i];
				if( lm->getRelatedFramePtr().front()->getID() == 0 ) {
					++n_lms_alive;
					mean_parallax += lm->getMaxParallax();
					if(lm->getMaxParallax() >= params_.map_update.thres_parallax){
						++n_lms_parallax_ok;
					}
				}
			}
			mean_parallax /= (float)n_lms_alive;

#ifdef RECORD_LANDMARK_STAT
	statcurr_landmark.avg_parallax = mean_parallax;
	statcurr_landmark.n_ok_parallax = n_lms_parallax_ok;
#endif

			if(mean_parallax > params_.keyframe_update.thres_mean_parallax*110000)
				initialization_done = true;
			
			if(initialization_done){ // lms_tracked_ 의 평균 parallax가 특정 값 이상인 경우, 초기화 끝. 
				// lms_tracked_를 업데이트한다. 
				system_flags_.flagVOInit = true;

				std::cout << "VO initialzed!\n";
			}
		}
	}
	else { // VO initialized. Do track the new image.

	}

	// 

	// Update statistics
	stat_.stats_landmark.push_back(statcurr_landmark);
	stat_.stats_frame.push_back(statcurr_frame);
	stat_.stats_execution.push_back(statcurr_execution);
	std::cout << "Statistics Updated. size: " << stat_.stats_landmark.size() << "\n";

	// Replace the 'frame_prev_' with 'frame_curr'
	frame_prev_ = frame_curr;

	// Notify a thread.

	mut_scale_estimator_->lock();
	*flag_do_ASR_ = true;
	mut_scale_estimator_->unlock();
	cond_var_scale_estimator_->notify_all();
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

void ScaleMonoVO::saveLandmarks(const LandmarkPtr& lm, bool verbose){
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

void ScaleMonoVO::saveFrames(const FramePtr& frame, bool verbose){
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