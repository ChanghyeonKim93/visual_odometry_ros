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
	// std::thread thread_scale_recovery_;
	// std::mutex mut_;
	// std::condition_variable convar_dataready_;

private:
	struct SystemFlags{
		bool flagFirstImageGot;
		bool flagVOInit;
		SystemFlags():flagFirstImageGot(false), flagVOInit(false) {};
	};

	struct AlgorithmParameters{
		struct FeatureTrackerParameters{
			float thres_error       = 20.0; // KLT error threshold
			float thres_bidirection = 1.0; // bidirection pixel error threshold
			uint32_t window_size    = 15;  // KLT window size 
			uint32_t max_level      = 6;   // KLT maximum pyramid level
		};
		struct FeatureExtractorParameters{
			uint32_t n_bins_u       = 18; // Bucket grid size u
			uint32_t n_bins_v       = 8; // Bucket grid size v
			float thres_fastscore   = 20.0; // FAST score threshold
			float radius            = 10.0; // NONMAX pixel threshold
		};
		struct MotionEstimatorParameters{
			
		};
		struct KeyframeUpdateParameters{
			float thres_parallax    = 1.0*D2R;
		};
		FeatureTrackerParameters   feature_tracker;
		FeatureExtractorParameters feature_extractor;
		MotionEstimatorParameters  motion_estimator;
		KeyframeUpdateParameters   keyframe_update;
	};
	AlgorithmParameters params_;

// For tracker
private:
	SystemFlags    system_flags_;
	FramePtr       frame_prev_;
	FramePtr       keyframe_;

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
	void updateKeyframe(const FramePtr& frame);

private:
	void runDataset(); // run algorithm

private:
	void loadCameraIntrinsic_KITTI_IMAGE0(const std::string& dir); // functions for loading yaml files.
	void loadCameraIntrinsic(const std::string& dir);
};



#endif