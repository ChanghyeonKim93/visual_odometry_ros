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
: cam_(nullptr) {
	std::cout << "Scale mono VO starts\n";
	
	flag_vo_initialized_ = false;

	// Set cam_
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
	rows = fs["Camera.height"];
	cols = fs["Camera.width"];

	float fx,fy,cx,cy;
	fx = fs["Camera.fx"];
	fy = fs["Camera.fy"];
	cx = fs["Camera.cx"];
	cy = fs["Camera.cy"];

	float k1,k2,k3,p1,p2;
	k1 = fs["Camera.k1"];
	k2 = fs["Camera.k2"];
	k3 = fs["Camera.k3"];
	p1 = fs["Camera.p1"];
	p2 = fs["Camera.p2"];

	cv::Mat cvK_tmp;
	cvK_tmp = cv::Mat(3,3,CV_32FC1);
	cvK_tmp.at<float>(0,0) = fx;
	cvK_tmp.at<float>(0,1) = 0.0f;
	cvK_tmp.at<float>(0,2) = cx;
	cvK_tmp.at<float>(1,0) = 0.0f;
	cvK_tmp.at<float>(1,1) = fy;
	cvK_tmp.at<float>(1,2) = cy;
	cvK_tmp.at<float>(2,0) = 0.0f;
	cvK_tmp.at<float>(2,1) = 0.0f;
	cvK_tmp.at<float>(2,2) = 1.0f;
	
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

	// 이미지를  undistort 한다. (KITTI 라서 할 필요 없음.)
	// cv::Mat img_undist;
	// cam_->undistort(img, img_undist);
	// frame_curr_->setImageAndTimestamp(img_undist, timestamp);
	frame_curr->setImageAndTimestamp(img, timestamp);

	// 생성된 frame은 저장한다.
	all_frames_.push_back(frame_curr);
	if(!flag_vo_initialized_){ // Not initialized yet.
		
		// frame_prev_ 이 가지고 있는 related_landmarks_을 img		
		if(1){ // lms_tracked_ 의 평균 parallax가 특정 값 이상인 경우, 초기화 끝. 

			// lms_tracked_를 업데이트한다. 
			flag_vo_initialized_ = true;

			std::cout << "VO initialzed!\n";
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
	frame_prev_ = frame_curr;
};
