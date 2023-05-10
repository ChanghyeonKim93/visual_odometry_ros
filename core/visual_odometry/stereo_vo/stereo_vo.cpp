#include "core/visual_odometry/stereo_vo/stereo_vo.h"

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
	std::cout << "Save all frames trajectory...\n";
	std::string filedir_frame_poses = "/home/kch/frame_poses.txt";
	std::ofstream of_frame_poses(filedir_frame_poses, std::ios::trunc);
    // of_frame_poses.open(filedir_frame_poses, std::ios::trunc);

    of_frame_poses.precision(4);
    of_frame_poses.setf(std::ios_base::fixed, std::ios_base::floatfield);
    if(of_frame_poses.is_open())
	{
		for(int j = 0; j < this->all_stframes_.size(); ++j)
		{
			of_frame_poses  << all_stframes_[j]->getLeft()->getID() << " "
							<< all_stframes_[j]->getLeft()->getPose()(0,0) << " " 
							<< all_stframes_[j]->getLeft()->getPose()(0,1) << " " 
							<< all_stframes_[j]->getLeft()->getPose()(0,2) << " " 
							<< all_stframes_[j]->getLeft()->getPose()(0,3) << " " 
							<< all_stframes_[j]->getLeft()->getPose()(1,0) << " " 
							<< all_stframes_[j]->getLeft()->getPose()(1,1) << " " 
							<< all_stframes_[j]->getLeft()->getPose()(1,2) << " " 
							<< all_stframes_[j]->getLeft()->getPose()(1,3) << " " 
							<< all_stframes_[j]->getLeft()->getPose()(2,0) << " " 
							<< all_stframes_[j]->getLeft()->getPose()(2,1) << " " 
							<< all_stframes_[j]->getLeft()->getPose()(2,2) << " " 
							<< all_stframes_[j]->getLeft()->getPose()(2,3) << std::endl;
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
		for(int j = 0; j < this->all_stkeyframes_.size(); ++j)
		{
			of_keyframe_poses   << all_stkeyframes_[j]->getLeft()->getID() << " "
								<< all_stkeyframes_[j]->getLeft()->getPose()(0,0) << " " 
								<< all_stkeyframes_[j]->getLeft()->getPose()(0,1) << " " 
								<< all_stkeyframes_[j]->getLeft()->getPose()(0,2) << " " 
								<< all_stkeyframes_[j]->getLeft()->getPose()(0,3) << " " 
								<< all_stkeyframes_[j]->getLeft()->getPose()(1,0) << " " 
								<< all_stkeyframes_[j]->getLeft()->getPose()(1,1) << " " 
								<< all_stkeyframes_[j]->getLeft()->getPose()(1,2) << " " 
								<< all_stkeyframes_[j]->getLeft()->getPose()(1,3) << " " 
								<< all_stkeyframes_[j]->getLeft()->getPose()(2,0) << " " 
								<< all_stkeyframes_[j]->getLeft()->getPose()(2,1) << " " 
								<< all_stkeyframes_[j]->getLeft()->getPose()(2,2) << " " 
								<< all_stkeyframes_[j]->getLeft()->getPose()(2,3) << std::endl;
		}
    }
    else {
        throw std::runtime_error("file_dir cannot be opened!");
    }
	std::cout << " DONE!\n";
	
	std::cout << "Stereo VO is terminated.\n";
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
	for(auto& lm : lms)	
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
	for(auto& stf : stframes)
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


void StereoVO::trackStereoImages(
    const cv::Mat& img_left, const cv::Mat& img_right, const double& timestamp)
{
	float THRES_SAMPSON  = params_.feature_tracker.thres_sampson;
	float THRES_PARALLAX = params_.map_update.thres_parallax;

	// Generate statistics
	AlgorithmStatistics::LandmarkStatistics  statcurr_landmark;
	AlgorithmStatistics::FrameStatistics     statcurr_frame;
	AlgorithmStatistics::KeyframeStatistics  statcurr_keyframe;
	AlgorithmStatistics::ExecutionStatistics statcurr_execution;
    
	// 이미지 undistort (KITTI는 할 필요 X)
    PoseSE3 T_lr;
    PoseSE3 T_rl;

    // Left and Right Cameras.
    CameraPtr cam_left;
    CameraPtr cam_right;
    
    timer::tic();
	cv::Mat img_left_undist, img_right_undist;
	if( system_flags_.flagDoUndistortion )
	{
        stereo_cam_->rectifyStereoImages(
            img_left, img_right, 
            img_left_undist, img_right_undist);

		img_left_undist.convertTo(img_left_undist, CV_8UC1);
		img_right_undist.convertTo(img_right_undist, CV_8UC1);

        T_lr = stereo_cam_->getRectifiedStereoPoseLeft2Right();
        T_rl = stereo_cam_->getRectifiedStereoPoseRight2Left();

        cam_left  = stereo_cam_->getRectifiedCamera();
        cam_right = stereo_cam_->getRectifiedCamera();

	}
	else
	{
		img_left.copyTo(img_left_undist);
		img_right.copyTo(img_right_undist);
        T_lr = stereo_cam_->getStereoPoseLeft2Right();
        T_rl = stereo_cam_->getStereoPoseRight2Left();
        cam_left  = stereo_cam_->getLeftCamera(); 
        cam_right = stereo_cam_->getRightCamera(); 
	}
    std::cout << colorcode::text_green << "Time [stereo undistort ]: " << timer::toc(0) << " [ms]\n" << colorcode::cout_reset;

    // Algorithm implementation
    // Make new stereo frame for current images.
	StereoFramePtr stframe_curr = std::make_shared<StereoFrame>(img_left_undist, img_right_undist, cam_left, cam_right, timestamp);
	this->saveStereoFrame(stframe_curr);

    // Algorithm
    if( this->system_flags_.flagFirstImageGot )
    {
        // 초기화가 된 상태에서 반복적 구동.
        // 이전 이미지 가져오기
        const cv::Mat& I0_left  = stframe_prev_->getLeft()->getImage();
        const cv::Mat& I0_right = stframe_prev_->getRight()->getImage();
        
        // 현재 이미지 가져오기
        const cv::Mat& I1_left  = stframe_curr->getLeft()->getImage();
        const cv::Mat& I1_right = stframe_curr->getRight()->getImage();

        // [1] 직전 tracking 결과를 가져옴
        StereoLandmarkTracking lmtrack_prev;
        lmtrack_prev.pts_l0 = stframe_prev_->getLeft()->getPtsSeen();
        lmtrack_prev.pts_r0 = stframe_prev_->getRight()->getPtsSeen();
        lmtrack_prev.n_pts  = lmtrack_prev.pts_l0.size();
        lmtrack_prev.pts_l1 = PixelVec(lmtrack_prev.n_pts); // tracking result
        lmtrack_prev.pts_r1 = PixelVec(lmtrack_prev.n_pts);
        lmtrack_prev.lms    = stframe_prev_->getLeft()->getRelatedLandmarkPtr();

        int n_pts_exist = lmtrack_prev.n_pts;
        
        // [2] 현재 stereo frame의 자세를 업데이트한다.
        // 이전 자세와 이전 업데이트 값을 가져온다. (constant velocity assumption)
        const PoseSE3& T_wp = stframe_prev_->getLeft()->getPose();
        const PoseSE3& T_pw = stframe_prev_->getLeft()->getPoseInv();
        const PoseSE3& dT_pc_prev = stframe_prev_->getLeft()->getPoseDiff01();

        PoseSE3 T_wc_prior = T_wp * dT_pc_prev;
        PoseSE3 T_cw_prior = geometry::inverseSE3_f(T_wc_prior);

        // [3] tracking을 위해, 'pts_l1' 'pts_r1' 에 대한 prior 계산.
        std::vector<float> patch_scale_prior(n_pts_exist);
        for(int i = 0; i < n_pts_exist; ++i)
        {
            const LandmarkPtr& lm = lmtrack_prev.lms[i]; 
            float scale_tmp = 1.0f;

            if( lm->isTriangulated() )
            {
                const Point& X = lm->get3DPoint();
                Point X_l1 = T_cw_prior.block<3,3>(0,0)*X + T_cw_prior.block<3,1>(0,3);
                Point X_r1 = T_rl.block<3,3>(0,0)*X_l1 + T_rl.block<3,1>(0,3);

                // warp to current estimate
                Point X_l0 = T_pw.block<3,3>(0,0) * X + T_pw.block<3,1>(0,3);
                scale_tmp = X_l0(2)/X_l1(2);
                
                // Project prior point.
                Pixel pt_l1 = cam_left->projectToPixel(X_l1);
                Pixel pt_r1 = cam_right->projectToPixel(X_r1);
                if( !cam_left->inImage(pt_l1) || !cam_right->inImage(pt_r1) 
                    || X_l1(2) < 0.1 || X_r1(2) < 0.1 )
                { 
                    // out of image.
                    lmtrack_prev.pts_l1[i] = lmtrack_prev.pts_l0[i];
                    lmtrack_prev.pts_r1[i] = lmtrack_prev.pts_r0[i];
                }
                else
                {
                    lmtrack_prev.pts_l1[i] = pt_l1;
                    lmtrack_prev.pts_r1[i] = pt_r1;
                }
            }
            else   
            {
                lmtrack_prev.pts_l1[i] = lmtrack_prev.pts_l0[i];
                lmtrack_prev.pts_r1[i] = lmtrack_prev.pts_r0[i];
            }

            patch_scale_prior[i] = scale_tmp;
        }

        /* Tracking order
            l0 -> l1
            l1 -> r1
            // r0 -> r1
            // check : r1 == r1
        */
        // [4] Track l0 --> l1 (lmtrack_l0l1)
        timer::tic();
        MaskVec mask_l0l1;
        this->tracker_->trackWithPrior(
            I0_left, I1_left,
            lmtrack_prev.pts_l0, params_.feature_tracker.window_size, params_.feature_tracker.max_level, params_.feature_tracker.thres_error,
            lmtrack_prev.pts_l1, mask_l0l1); // lmtrack_curr.pts_u1 에 prior pixels가 이미 들어있다.

        StereoLandmarkTracking lmtrack_l0l1_notrefine(lmtrack_prev, mask_l0l1);
        std::cout << "# pts : " << lmtrack_l0l1_notrefine.n_pts << std::endl;
		std::cout << colorcode::text_green << "Time [track l0l1]: " << timer::toc(0) << " [ms]\n" << colorcode::cout_reset;       

        // [4-1] track with scale
        timer::tic();
		MaskVec mask_refine(lmtrack_l0l1_notrefine.pts_l0.size(), true);
		const cv::Mat& du0_left = stframe_prev_->getLeft()->getImageDu();
		const cv::Mat& dv0_left = stframe_prev_->getLeft()->getImageDv();
		tracker_->trackWithScale(
			I0_left, du0_left, dv0_left, I1_left, 
			lmtrack_l0l1_notrefine.pts_l0, patch_scale_prior, lmtrack_l0l1_notrefine.pts_l1,
			mask_refine); 

		StereoLandmarkTracking lmtrack_l0l1(lmtrack_l0l1_notrefine, mask_refine);
		std::cout << colorcode::text_green << "Time [trackWithScale   ]: " << timer::toc(0) << " [ms]\n" << colorcode::cout_reset;
		

        // [5] Track l1 --> r1 (lmtrack_l1r1)
        timer::tic();
        MaskVec mask_l1r1;
        this->tracker_->trackWithPrior(
            I1_left, I1_right,
            lmtrack_l0l1.pts_l1, params_.feature_tracker.window_size, params_.feature_tracker.max_level, params_.feature_tracker.thres_error,
            lmtrack_l0l1.pts_r1, mask_l1r1); // lmtrack_curr.pts_u1 에 prior pixels가 이미 들어있다.

        StereoLandmarkTracking lmtrack_kltok(lmtrack_l0l1, mask_l1r1);
        std::cout << "# pts : " << lmtrack_kltok.n_pts << std::endl;
		std::cout << colorcode::text_green << "Time [track l1r1]: " << timer::toc(0) << " [ms]\n" << colorcode::cout_reset;

        // [6] Motion Estimation via pose-only BA (stereo version)
        // Using landmarks with 3D point, stereo pose-only BA.
        // pts_left_1 , pts_right_1 , Xw , T_cw_prior    : needed. 
        //    --> return: T_wc
        timer::tic();
        MaskVec mask_motion(lmtrack_kltok.pts_l1.size(), true); 

        // Find point with 3D points
        PoseSE3 dT_pc_poBA = dT_pc_prev;

        PoseSE3 T_wc = T_wc_prior;

        int cnt_poBA = 0;
        PointVec Xp_poBA;
        PixelVec pts_l1_poBA;
        PixelVec pts_r1_poBA;
        MaskVec mask_poBA;
        std::vector<int> index_poBA;
        for(int i = 0; i < lmtrack_kltok.pts_l1.size(); ++i)
        {
            LandmarkConstPtr& lm = lmtrack_kltok.lms[i];

            if( lm->isTriangulated() )
            {   
                const Pixel& pt_l1 = lmtrack_kltok.pts_l1[i];
                const Pixel& pt_r1 = lmtrack_kltok.pts_r1[i];
                const Point& X     = lm->get3DPoint();

                Point Xp = T_pw.block<3,3>(0,0) * X + T_pw.block<3,1>(0,3);

                Xp_poBA.push_back(Xp);
                pts_l1_poBA.push_back(pt_l1);
                pts_r1_poBA.push_back(pt_r1);
                mask_poBA.push_back(true);
                index_poBA.push_back(i);
                ++cnt_poBA;
            }
        }
        std::cout << "  # of poseonly BA point: " << cnt_poBA << std::endl;

        bool poseonlyBA_success =
            motion_estimator_->poseOnlyBundleAdjustment_Stereo(
                Xp_poBA, pts_l1_poBA, pts_r1_poBA, cam_left, cam_right, T_lr, params_.motion_estimator.thres_poseba_error,
                dT_pc_poBA, mask_poBA);

        if( !poseonlyBA_success )
        {
            throw std::runtime_error("PoseOnlyStereoBA is failed!");
        }
        else{
            // Success. Update pose.
            for(int i = 0; i < mask_poBA.size(); ++i)
            {
                const int& idx = index_poBA[i];
                if( mask_poBA[i] )
                    mask_motion[idx] = true;
                else 
                    mask_motion[idx] = false;
            }

            T_wc = T_wp * dT_pc_poBA;

            stframe_curr->setStereoPoseByLeft(T_wc, T_lr);
            stframe_curr->getLeft()->setPoseDiff10(dT_pc_poBA.inverse());
        }

        StereoLandmarkTracking lmtrack_motion_ok(lmtrack_kltok, mask_motion);
		std::cout << colorcode::text_green << "Time [motion est]: " << timer::toc(0) << " [ms]\n" << colorcode::cout_reset;


        // [7] Check sampson distance 0.01 ms
        timer::tic();
        std::vector<float> symm_epi_dist;
        symm_epi_dist.resize(lmtrack_motion_ok.pts_l1.size(), 0);
        // motion_estimator_->calcSymmetricEpipolarDistance(lmtrack_motion_ok.pts_l0, lmtrack_motion_ok.pts_l1, cam_rect, dT_pc_poBA.block<3,3>(0,0), dT_pc_poBA.block<3,1>(0,3), symm_epi_dist);
        for(int i = 0; i < lmtrack_motion_ok.pts_l1.size(); ++i)
        {
            // std::cout << i << "-th point epi dist: " << symm_epi_dist[i] << std::endl;
            if( lmtrack_motion_ok.pts_l1[i].y > 660)
                symm_epi_dist[i] = 100;
            // else  
                // symm_epi_dist[i] = 0;
        }

        MaskVec mask_sampson(lmtrack_motion_ok.pts_l1.size(), true);
        for(int i = 0; i < mask_sampson.size(); ++i)
            mask_sampson[i] = (symm_epi_dist[i] < THRES_SAMPSON);
        
        StereoLandmarkTracking lmtrack_final(lmtrack_motion_ok, mask_sampson);
        std::cout << "# pts : " << lmtrack_final.n_pts << std::endl;
		std::cout << colorcode::text_green << "Time [sampson ]: " << timer::toc(0) << " [ms]\n" << colorcode::cout_reset;




        // [8] Update observations for surviving landmarks
        for(int i = 0; i < lmtrack_final.pts_l1.size(); ++i)
        {
            const LandmarkPtr& lm = lmtrack_final.lms[i];
            lm->addObservationAndRelatedFrame(lmtrack_final.pts_l1[i], stframe_curr->getLeft());
            lm->addObservationAndRelatedFrame(lmtrack_final.pts_r1[i], stframe_curr->getRight());
        }  

        if(1)
        {
            this->showTrackingBA( "stereo_tracking", I1_left, PixelVec(), lmtrack_final.pts_l1);
        }

        // [10] Extract new points from empty bins.
        PixelVec pts_l1_new;
        extractor_->updateWeightBin(lmtrack_final.pts_l1);
        extractor_->extractORBwithBinning_fast(I1_left, pts_l1_new, true);

        int n_pts_new = pts_l1_new.size();
        std::cout << "# of newly extracted pts on empty space: " << n_pts_new << std::endl; 

        if(n_pts_new > 0)
        {
            // If there are new points, add it.
            std::cout << " --- --- # of NEW points : " << n_pts_new  << "\n";

            // Track static stereo
            MaskVec mask_new;
            PixelVec pts_r1_new;
            this->tracker_->trackBidirection(
                I1_left, I1_right, pts_l1_new, 
                params_.feature_tracker.window_size, params_.feature_tracker.max_level, params_.feature_tracker.thres_error, params_.feature_tracker.thres_bidirection,
                pts_r1_new, mask_new); // lmtrack_curr.pts_u1 에 prior pixels가 이미 들어있다.

            // 새로운 특징점은 새로운 landmark가 된다.
            for(int i = 0; i < pts_r1_new.size(); ++i)
            {
                if( mask_new[i] )
                {
                    const Pixel& pt_l1_new = pts_l1_new[i];
                    const Pixel& pt_r1_new = pts_r1_new[i];

                    // Reconstruct 3D point
                    Point Xl, Xr;
                    mapping::triangulateDLT(pt_l1_new, pt_r1_new, T_rl.block<3,3>(0,0), T_rl.block<3,1>(0,3), cam_left, cam_right, Xl, Xr);

                    if( Xl(2) > 0 && Xr(2) > 0)
                    {
                        // Point Xw = T_wc.block<3,3>(0,0) * Xl + T_wc.block<3,1>(0,3);
                    
                        LandmarkPtr lmptr = std::make_shared<Landmark>(pt_l1_new, stframe_curr->getLeft(), cam_left);
                        lmptr->addObservationAndRelatedFrame(pt_r1_new, stframe_curr->getRight());

                        lmtrack_final.pts_l1.push_back(pt_l1_new);
                        lmtrack_final.pts_r1.push_back(pt_r1_new);
                        lmtrack_final.lms.push_back(lmptr);

                        // lmptr->set3DPoint(Xw);
                    }
                    
                }
            }
        }
        else
            std::cout << " --- --- NO NEW POINTS.\n";
        

        // [11] Update tracking information (set related pixels and landmarks)
        std::cout << "# of final landmarks : " << lmtrack_final.pts_l1.size() << std::endl;
        stframe_curr->setStereoPtsSeenAndRelatedLandmarks(lmtrack_final.pts_l1, lmtrack_final.pts_r1, lmtrack_final.lms);

        // [12] Keyframe selection?
        bool flag_add_new_keyframe = this->stkeyframes_->checkUpdateRule(stframe_curr);

        if( flag_add_new_keyframe )
        {
            // Add keyframe
            this->saveStereoKeyframe(stframe_curr);
            this->stkeyframes_->addNewStereoKeyframe(stframe_curr);     

            // Reconstruction new features.
            int cnt_recon = 0;
            const Rot3& R_rl = T_rl.block<3,3>(0,0);
            const Pos3& t_rl = T_rl.block<3,1>(0,3);
            for(int i = 0; i < lmtrack_final.n_pts; ++i)
            {
                const Pixel&      pt0 = lmtrack_final.pts_l1[i];
                const Pixel&      pt1 = lmtrack_final.pts_r1[i];
                const LandmarkPtr& lm = lmtrack_final.lms[i];
                
                // Reconstruct points
                Point Xl, Xr;
                mapping::triangulateDLT(pt0, pt1, R_rl, t_rl, cam_left, cam_right, Xl, Xr);

                // Check reprojection error for the first image
                Pixel pt0_proj = cam_left->projectToPixel(Xl);
                Pixel dpt0 = pt0 - pt0_proj;
                float dpt0_norm2 = dpt0.x*dpt0.x + dpt0.y*dpt0.y;
                if(dpt0_norm2 > 1.0) continue;

                Pixel pt1_proj = cam_right->projectToPixel(Xr);
                Pixel dpt1 = pt1 - pt1_proj;
                float dpt1_norm2 = dpt1.x*dpt1.x + dpt1.y*dpt1.y;
                if(dpt1_norm2 > 1.0) continue;

                // Check the point in front of cameras
                if(Xl(2) > 0 && Xr(2) > 0) 
                {
                    Point Xworld = T_wc.block<3,3>(0,0)* Xl  + T_wc.block<3,1>(0,3);
                    lm->set3DPoint(Xworld);
                    ++cnt_recon;
                }
            }
            std::cout << "# of reconstructed points: " << cnt_recon << " / " << lmtrack_final.n_pts << std::endl;
            

            // Local Bundle Adjustment
            motion_estimator_->localBundleAdjustmentSparseSolver_Stereo(stkeyframes_, cam_left, cam_right, T_lr);

timer::tic();
PointVec X_tmp;
const LandmarkPtrVec& lmvec_tmp = stframe_curr->getLeft()->getRelatedLandmarkPtr();
for(int i = 0; i < lmvec_tmp.size(); ++i)
{
	X_tmp.push_back(lmvec_tmp[i]->get3DPoint());
}
statcurr_keyframe.Twc = stframe_curr->getLeft()->getPose();
statcurr_keyframe.mappoints = X_tmp;
stat_.stats_keyframe.push_back(statcurr_keyframe);

for(int j = 0; j < stat_.stats_keyframe.size(); ++j){
	stat_.stats_keyframe[j].Twc = all_stkeyframes_[j]->getLeft()->getPose();

	const LandmarkPtrVec& lmvec_tmp = all_stkeyframes_[j]->getLeft()->getRelatedLandmarkPtr();
	stat_.stats_keyframe[j].mappoints.resize(lmvec_tmp.size());
	for(int i = 0; i < lmvec_tmp.size(); ++i) {
		stat_.stats_keyframe[j].mappoints[i] = lmvec_tmp[i]->get3DPoint();
	}
}
std::cout << colorcode::text_green << "Time [RECORD KEYFR STAT]: " << timer::toc(0) << " [ms]\n" << colorcode::cout_reset;

        } // KEYFRAME addition done.


    }
    else 
    {
        /*
        ===================================================================
        ===================================================================
        ===================================================================
        ===================================================================
        ===================== The very first image ========================
        ===================================================================
        ===================================================================
        ===================================================================
        ===================================================================
        */
        const cv::Mat& I1_left  = stframe_curr->getLeft()->getImage();
        const cv::Mat& I1_right = stframe_curr->getRight()->getImage();

        // 첫번째 stereo frame의 자세는 아래와 같다. 
        stframe_curr->setStereoPoseByLeft(PoseSE3::Identity(), T_lr);
        stframe_curr->getLeft()->setPoseDiff10(PoseSE3::Identity());
        stframe_curr->getRight()->setPoseDiff10(PoseSE3::Identity());

        
        // Extract initial feature points.
        timer::tic();
        StereoLandmarkTracking lmtrack_curr;
        extractor_->resetWeightBin();
        extractor_->extractORBwithBinning_fast(I1_left, lmtrack_curr.pts_l1, true);
        lmtrack_curr.n_pts = lmtrack_curr.pts_l1.size();
        std::cout << "# extracted features : " << lmtrack_curr.n_pts << std::endl;
		std::cout << colorcode::text_green << "Time [extract ]: " << timer::toc(0) << " [ms]\n" << colorcode::cout_reset;       

        // make dummy pixels
        lmtrack_curr.pts_l0.resize(lmtrack_curr.n_pts);
        lmtrack_curr.pts_r0.resize(lmtrack_curr.n_pts);
        lmtrack_curr.pts_r1.resize(lmtrack_curr.n_pts);
        lmtrack_curr.lms.resize(lmtrack_curr.n_pts);

        // Feature tracking.
        // 1) Track 'static stereo' (I1_left --> I1_right)
        timer::tic();
        MaskVec mask_track(lmtrack_curr.n_pts, true);
        this->tracker_->trackBidirection(I1_left, I1_right,
            lmtrack_curr.pts_l1, params_.feature_tracker.window_size, params_.feature_tracker.max_level, params_.feature_tracker.thres_error, params_.feature_tracker.thres_bidirection,
            lmtrack_curr.pts_r1, mask_track); // lmtrack_curr.pts_u1 에 prior pixels가 이미 들어있다.

        StereoLandmarkTracking lmtrack_staticklt;
        for(int i = 0; i < lmtrack_curr.pts_l1.size(); ++i)
        {
            if( mask_track[i] )
            {
                lmtrack_staticklt.pts_l1.push_back(lmtrack_curr.pts_l1[i]);
                lmtrack_staticklt.pts_r1.push_back(lmtrack_curr.pts_r1[i]);
            }
        }
        lmtrack_staticklt.n_pts = lmtrack_staticklt.pts_l1.size();
        lmtrack_staticklt.pts_l0.resize(lmtrack_staticklt.n_pts);
        lmtrack_staticklt.pts_r0.resize(lmtrack_staticklt.n_pts);
        lmtrack_staticklt.lms.resize(lmtrack_staticklt.n_pts);

		std::cout << colorcode::text_green << "Time [track bidirection]: " << timer::toc(0) << " [ms]\n" << colorcode::cout_reset;       

        // make landmarks
        for(int i = 0; i < lmtrack_staticklt.n_pts; ++i)
        {
            const Pixel& pt_left  = lmtrack_staticklt.pts_l1[i];
            const Pixel& pt_right = lmtrack_staticklt.pts_r1[i];
            LandmarkPtr lmptr = std::make_shared<Landmark>(pt_left, stframe_curr->getLeft(), cam_left);
            lmptr->addObservationAndRelatedFrame(pt_right, stframe_curr->getRight());

            lmtrack_staticklt.lms[i] = lmptr;

            this->saveLandmark(lmptr);                        
        }
        std::cout << "# static klt success pts : " << lmtrack_staticklt.n_pts  << std::endl;

        // 3D reconstruction 
        int cnt_recon = 0;
        const Rot3& R_rl = T_rl.block<3,3>(0,0);
        const Pos3& t_rl = T_rl.block<3,1>(0,3);
        for(int i = 0; i < lmtrack_staticklt.n_pts; ++i)
        {
            const Pixel&      pt0 = lmtrack_staticklt.pts_l1[i];
            const Pixel&      pt1 = lmtrack_staticklt.pts_r1[i];
            const LandmarkPtr& lm = lmtrack_staticklt.lms[i];
            
            // Reconstruct points
            Point Xl, Xr;
            mapping::triangulateDLT(pt0, pt1, R_rl, t_rl, cam_left, cam_right, Xl, Xr);

            // Check reprojection error for the first image
            Pixel pt0_proj = cam_left->projectToPixel(Xl);
            Pixel dpt0 = pt0 - pt0_proj;
            float dpt0_norm2 = dpt0.x*dpt0.x + dpt0.y*dpt0.y;
            if(dpt0_norm2 > 1.0) continue;

            Pixel pt1_proj = cam_right->projectToPixel(Xr);
            Pixel dpt1 = pt1 - pt1_proj;
            float dpt1_norm2 = dpt1.x*dpt1.x + dpt1.y*dpt1.y;
            if(dpt1_norm2 > 1.0) continue;

            // Check the point in front of cameras
            if(Xl(2) > 0 && Xr(2) > 0) 
            {
                Point Xworld = Xl;
                lm->set3DPoint(Xworld);
                ++cnt_recon;
            }
        }
        std::cout << "# of reconstructed points: " << cnt_recon << " / " << lmtrack_staticklt.n_pts << std::endl;
        
        // Set related pixels and landmarks    
        stframe_curr->getLeft()->setPtsSeenAndRelatedLandmarks(lmtrack_staticklt.pts_l1, lmtrack_staticklt.lms);
        stframe_curr->getRight()->setPtsSeenAndRelatedLandmarks(lmtrack_staticklt.pts_r1, lmtrack_staticklt.lms);


        this->system_flags_.flagFirstImageGot = true;

        std::cout << "============ End initialization. Start to iterate all images... ============" << std::endl;
        

        if(0) 
        {
            cv::Mat img_color;
            I1_right.convertTo(img_color, CV_8UC1);
            cv::cvtColor(img_color, img_color, CV_GRAY2RGB);
            for(const auto& pt : lmtrack_staticklt.pts_r1)
                cv::circle(img_color, pt, 5, cv::Scalar(0,255,0));
            
            std::stringstream ss;
            ss << " - feature_backtrack";
            cv::namedWindow(ss.str());
            cv::imshow(ss.str(), img_color);


            I1_left.convertTo(img_color, CV_8UC1);
            cv::cvtColor(img_color, img_color, CV_GRAY2RGB);
            for(const auto& pt : lmtrack_staticklt.pts_l1)
                cv::circle(img_color, pt, 5, cv::Scalar(0,255,0));
            
            ss;
            ss << " - feature_track";
            cv::namedWindow(ss.str());
            cv::imshow(ss.str(), img_color);
            cv::waitKey(0);
        }
        

    }
    // Update statistics
	stat_.stats_frame.push_back(statcurr_frame);
    stat_.stats_frame.back().Twc = all_stframes_.back()->getLeft()->getPose();

// [6] Update prev
	this->stframe_prev_ = stframe_curr;	

	std::cout << "# of all landmarks: " << all_landmarks_.size() << std::endl;

};
