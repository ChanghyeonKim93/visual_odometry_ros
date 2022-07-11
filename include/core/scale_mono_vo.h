#ifndef _SCALE_MONO_VO_H_
#define _SCALE_MONO_VO_H_

#include <iostream>
#include <exception>
#include <string>

#include <thread>
#include <mutex>
#include <condition_variable>

#include <ros/ros.h>

// Eigen
#include <Eigen/Dense>

// OpenCV
#include <cv_bridge/cv_bridge.h>

#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// Defines 
#include "core/defines.h"
#include "core/type_defines.h"

// custom
#include "core/camera.h"

#include "core/frame.h"
#include "core/landmark.h"

#include "core/feature_extractor.h"
#include "core/feature_tracker.h"
#include "core/motion_estimator.h"

#include "core/image_processing.h"
#include "core/dataset_loader.h"

#include "util/timer.h"

class ScaleMonoVO;

class ScaleMonoVO {
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

// Dataset related.
private:
	dataset_loader::DatasetStruct dataset_;

// Camera object
private:
	std::shared_ptr<Camera>           cam_;

// Modules
private:
	std::shared_ptr<FeatureExtractor> extractor_;
	std::shared_ptr<FeatureTracker>   tracker_;
	std::shared_ptr<MotionEstimator>  motion_estimator_;

// For scale recovery thread
private:
	std::thread thread_scale_recovery_;
	std::mutex mut_;
	std::condition_variable convar_dataready_;

private:
	struct SystemFlags{
		bool flagFirstImageGot;
		bool flagVOInit;
	};

// For tracker
private:
	SystemFlags system_flags_;
	FramePtr       frame_prev_;
	
// All frames and landmarks
private:
	
	LandmarkPtrVec all_landmarks_;
	FramePtrVec    all_frames_;
	FramePtrVec    all_keyframes_;
	
public:
	ScaleMonoVO(std::string mode, std::string directory_intrinsic);
	~ScaleMonoVO();

	void trackImage(const cv::Mat& img, const double& timestamp);

private:
	void runDataset(); // run algorithm

private:
	void loadCameraIntrinsic_KITTI_IMAGE0(const std::string& dir); // functions for loading yaml files.
	void loadCameraIntrinsic(const std::string& dir);
};



#endif