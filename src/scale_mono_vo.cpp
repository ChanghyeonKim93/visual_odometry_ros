#include "scale_mono_vo.h"

ScaleMonoVO::ScaleMonoVO(std::string mode) {
	std::cout << "Scale mono VO starts\n";

	std::string dir_dataset = "D:/#DATASET/kitti/data_odometry_gray";
	std::string dataset_num = "00";

	// Set cam_
	cam_ = std::make_shared<Camera>();
	this->loadCameraIntrinsic_KITTI_IMAGE0(dir_dataset + "/dataset/sequences/" + dataset_num + "/intrinsic.yaml");

	// Get dataset filenames.
	dataset_loader::getImageFileNames_KITTI(dir_dataset, dataset_num, dataset);

	if(mode == "dataset"){
		std::cout <<"ScaleMonoVO - 'dataset' mode.\n";
		// run while loop
		runDataset();
	}
	else if(mode == "rosbag"){
		std::cout <<"ScaleMonoVO - 'rosbag' mode.\n";
		// wait callback ...
	}
	else throw std::runtime_error("ScaleMonoVO - unknown mode.");

};

ScaleMonoVO::~ScaleMonoVO() {
	std::cout << "Scale mono VO is terminated.\n";
};

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


void ScaleMonoVO::trackMonocular(const cv::Mat& img, const double& timestamp) {

	// Make a new frame
	FramePtr frame_cur = std::make_shared<Frame>();
	frame_cur->setImage(img);
	// frame_current->setPose();
	// frame_cur->setLandmarks();

	//


	// cv::Mat Tcw = motion_tracker_->trackCurrentImage(img, timestamp);

    // mTrackingState = mpTracker->mState;
    // mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
    // mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;

	// current_position_ = Tcw;
};


void ScaleMonoVO::runDataset() {
	
	cv::Mat img0 = cv::imread(dataset.image_names[0], cv::IMREAD_GRAYSCALE);
	cv::Mat img0f;
	img0.convertTo(img0f, CV_32FC1);
	std::cout << "input image type: " << image_processing::type2str(img0f) << std::endl;
	/*while (true) {

	}*/
};