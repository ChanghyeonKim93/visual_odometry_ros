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
: cam_(nullptr), system_flags_() {
	std::cout << "Scale mono VO starts\n";
	
	system_flags_.flagFirstImageGot  = false;
	system_flags_.flagVOInit         = false;

	// Initialize camera
	cam_ = std::make_shared<Camera>();

	if(mode == "dataset"){
		std::cout <<"ScaleMonoVO - 'dataset' mode.\n";
		std::string dir_dataset = "D:/#DATASET/kitti/data_odometry_gray";
		std::string dataset_num = "00";

		this->loadCameraIntrinsic_KITTI_IMAGE0(dir_dataset + "/dataset/sequences/" + dataset_num + "/intrinsic.yaml");

		// Get dataset filenames.
		dataset_loader::getImageFileNames_KITTI(dir_dataset, dataset_num, dataset_);

		runDataset(); // run while loop
	}
	else if(mode == "rosbag"){
		std::cout << "ScaleMonoVO - 'rosbag' mode.\n";
		
		this->loadCameraIntrinsic(directory_intrinsic);
		// wait callback ...
	}
	else 
		throw std::runtime_error("ScaleMonoVO - unknown mode.");

	// Initialize feature extractor (ORB-based)
	extractor_ = std::make_shared<FeatureExtractor>();
	int n_bins_u = 20;
	int n_bins_v = 12;
	float THRES_FAST = 20.0;
	float radius = 10.0;
	extractor_->initParams(cam_->cols(), cam_->rows(), n_bins_u, n_bins_v, THRES_FAST, radius);

	// Initialize feature tracker (KLT-based)
	tracker_ = std::make_shared<FeatureTracker>();

	// Initialize motion estimator
	motion_estimator_ = std::make_shared<MotionEstimator>();

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
 * @date 11-July-2022
 */
void ScaleMonoVO::loadCameraIntrinsic_KITTI_IMAGE0(const std::string& dir) {

	cv::FileStorage fs(dir, cv::FileStorage::READ);
	if (!fs.isOpened()) throw std::runtime_error("intrinsic file cannot be found!\n");

	int rows_tmp, cols_tmp;

	cv::Mat cvK_tmp, cvD_tmp;
	rows_tmp = fs["cam0.height"];
	cols_tmp = fs["cam0.width"];
	fs["cam0.K"] >> cvK_tmp;
	fs["cam0.D"] >> cvD_tmp;
	cam_->initParams(cols_tmp, rows_tmp, cvK_tmp, cvD_tmp);
	
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
void ScaleMonoVO::loadCameraIntrinsic(const std::string& dir) {
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
	// 현재 들어온 이미지에 대한 Frame을 생성한다.
	FramePtr frame_curr = std::make_shared<Frame>();

	// 이미지를 undistort 한다. (KITTI라서 할 필요 없음.)
	bool flag_do_undistort = false;
	cv::Mat img_undist;
	if(flag_do_undistort) cam_->undistort(img, img_undist);
	else img.copyTo(img_undist);

	frame_curr->setImageAndTimestamp(img_undist, timestamp);
	
	// 생성된 frame은 저장한다.
	if( !system_flags_.flagVOInit ) { // 아직 초기화가 되지 않았다.
		if( !system_flags_.flagFirstImageGot ) { // 최초 첫 이미지가 아직 안들어온 상태.
			// Get the first image.
			const cv::Mat& I0 = frame_curr->getImage();

			// Extract pixels
			PixelVec       pxvec0;
			LandmarkPtrVec lmvec0;

			extractor_->resetWeightBin();
			extractor_->extractORBwithBinning(I0, pxvec0);
			
			// Set initial landmarks
			lmvec0.reserve(pxvec0.size());
			for(auto p : pxvec0) lmvec0.push_back(std::make_shared<Landmark>(p, frame_curr));
			
			frame_curr->setRelatedLandmarks(lmvec0);
			frame_curr->setPtsSeen(pxvec0);

			if(1){
				cv::namedWindow("img_features");
				cv::Mat img_draw;
				frame_curr->getImage().copyTo(img_draw);
				cv::cvtColor(img_draw,img_draw, CV_GRAY2RGB);
				for(auto p : pxvec0)
					cv::circle(img_draw, p, 4.0, cv::Scalar(255,0,255));
				
				cv::imshow("img_features", img_draw);
				cv::waitKey(3);
			}

			system_flags_.flagFirstImageGot = true;
			std::cout << "The first image is got.\n";
		}
		else { // 최초 첫 이미지는 들어왔으나, 아직 초기화가 되지 않은 상태.
			   // 초기화는 맨 첫 이미지 (첫 키프레임) 대비, 제대로 추적 된 landmark가 60 퍼센트 이상이며, 
			   // 추적된 landmark 각각의 최대 parallax 가 1도 이상인 경우 초기화 완료.
			// 이전 프레임의 pixels 와 lms0를 가져온다.
			const PixelVec&       pxvec0 = frame_prev_->getPtsSeen();
			const LandmarkPtrVec& lmvec0 = frame_prev_->getRelatedLandmarkPtr();
			
			// frame_prev_ 의 lms 를 현재 이미지로 track.
			float thres_err         = 20.0;
			float thres_bidirection = 1.0;

			PixelVec pxvec1_track;
			MaskVec maskvec1_track;
			tracker_->trackBidirection(frame_prev_->getImage(), frame_curr->getImage(), pxvec0, thres_err, thres_bidirection,
							pxvec1_track, maskvec1_track);

			// Tracking 결과를 반영하여 pxvec1_alive, lmvec1_alive를 정리한다.
			LandmarkPtrVec lmvec1_alive;
			PixelVec       pxvec0_alive;
			PixelVec       pxvec1_alive;
			for(int i = 0; i < pxvec1_track.size(); ++i){
				if( maskvec1_track[i]){
					pxvec0_alive.push_back(pxvec0[i]);
					lmvec1_alive.push_back(lmvec0[i]);
					pxvec1_alive.push_back(pxvec1_track[i]);
				}
				else lmvec0[i]->setAlive(false); // track failed. Dead point.
			}
			
			// pts0 와 pts1을 이용, 5-point algorithm 으로 모션을 구한다.
			MaskVec maskvec_inlier;
			motion_estimator_->calcPose5PointsAlgorithm(pxvec0_alive, pxvec1_alive, cam_, maskvec_inlier);
			
			// tracking, 5p algorithm, newpoint 모두 합쳐서 살아남은 점만 frame_curr에 넣는다
			LandmarkPtrVec lmvec1_final;
			PixelVec       pxvec1_final;
			for(int i = 0; i < pxvec1_alive.size(); ++i){
				if( maskvec_inlier[i] ) {
					lmvec1_alive[i]->addObservationAndRelatedFrame(pxvec1_alive[i], frame_curr);
					lmvec1_final.push_back(lmvec1_alive[i]);
					pxvec1_final.push_back(pxvec1_alive[i]);
				}
				else lmvec1_alive[i]->setAlive(false); // 5p algorithm failed. Dead point.
			}			

			// 빈 곳에 특징점 pts1_new 를 추출한다.
			PixelVec pxvec1_new;
			extractor_->updateWeightBin(pxvec1_final); // 이미 pts1가 있는 곳은 제외.
			extractor_->extractORBwithBinning(frame_curr->getImage(), pxvec1_new);

			if(!pxvec1_new.empty()){
				// 새로운 특징점은 새로운 landmark가 된다.
				for(auto p1_new : pxvec1_new) {
					lmvec1_final.push_back(std::make_shared<Landmark>(p1_new, frame_curr));
					pxvec1_final.emplace_back(p1_new);
				}
				std::cout << "pts1_new size: " << pxvec1_new.size() << std::endl;
			}


			// lms1와 pts1을 frame_curr에 넣는다.
			frame_curr->setRelatedLandmarks(lmvec1_final);
			frame_curr->setPtsSeen(pxvec1_final);

			if( true ){
				cv::namedWindow("img_features");
				cv::Mat img_draw;
				frame_curr->getImage().copyTo(img_draw);
				cv::cvtColor(img_draw,img_draw, CV_GRAY2RGB);
				for(int i = 0; i < pxvec0.size(); ++i) {
					if(maskvec1_track[i]) cv::circle(img_draw, pxvec0[i], 4.0, cv::Scalar(255,0,255)); // alived magenta
					else cv::circle(img_draw, pxvec0[i], 2.0, cv::Scalar(0,0,255)); // red, dead points
				}
				for(int i = 0; i < pxvec1_final.size(); ++i)
					cv::circle(img_draw, pxvec1_final[i], 4.0, cv::Scalar(0,255,0)); // green tracked points
				for(int i = 0; i < pxvec1_new.size(); ++i)
					cv::circle(img_draw, pxvec1_new[i], 4.0, cv::Scalar(255,128,0)); // blue new points
				
				cv::imshow("img_features", img_draw);
				cv::waitKey(5);
			}

			if(false){ // lms_tracked_ 의 평균 parallax가 특정 값 이상인 경우, 초기화 끝. 
				// lms_tracked_를 업데이트한다. 
				system_flags_.flagVOInit = true;

				std::cout << "VO initialzed!\n";
			}
		}		
	}	
	else { // VO initialized. Do track the new image.
		const double dt = timestamp - frame_prev_->getTimestamp();
		std::cout << "dt_img: " << dt << " sec." << std::endl;

		// i-2, i-1 째 이미지를 이용해서 motion velocity를 구한다. 
		// Eigen::Matrix4f dTdt;
		// dTdt = T(i-2)^-1*T(i-1) / (t(i-1) - t(i-2))
		// dTdt*dt;
		// Eigen::Matrix4f Twc_prior = frame_prev_->getPose()*(dTdt*(float)dt);

		// lms_tracked_ 를 가져온다.
		// lms_tracked_ 중, depth가 있는 점에 대해서 prior 계산한다.

	}

	// Add the newly incoming frame into the frame stack
	all_frames_.push_back(frame_curr);

	// Replace the 'frame_prev_' with 'frame_curr'
	frame_prev_ = frame_curr;
};
