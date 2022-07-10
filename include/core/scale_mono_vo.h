#ifndef _SCALE_MONO_VO_H_
#define _SCALE_MONO_VO_H_

#include <iostream>
#include <exception>
#include <string>

#include <thread>
#include <mutex>
#include <condition_variable>

#include <ros/ros.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/image_encodings.h>
#include <image_transport/image_transport.h>

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

// custom
#include "core/camera.h"

#include "core/frame.h"
#include "core/landmark.h"

#include "core/feature_extractor.h"
#include "core/image_processing.h"
#include "core/dataset_loader.h"

#include "util/timer.h"

class ScaleMonoVO
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

// dataset related.
private:
	dataset_loader::DatasetStruct dataset_;

private:
	std::shared_ptr<Camera> cam_;
	std::shared_ptr<FeatureExtractor> extractor_;
	//std::shared_ptr<FeatureTracker>   tracker_;
	//std::shared_ptr<MotionTracker>    motion_tracker_;

// For scale recovery thread
private:
	std::thread thread_scale_recovery_;
	std::mutex mut_;
	std::condition_variable convar_dataready_;

// For tracker
private:
	bool flag_vo_initialized_;

	FramePtr frame_prev_;
	std::vector<LandmarkPtr> lms_prev_;
	
// All frames and landmarks
private:
	std::vector<LandmarkPtr> all_landmarks_;
	std::vector<FramePtr>    all_frames_;
	std::vector<FramePtr>    all_keyframes_;
	
public:
	ScaleMonoVO(std::string mode);
	~ScaleMonoVO();

	void trackImage(const cv::Mat& img, const double& timestamp);

private:
	void runDataset(); // run algorithm

private:
	void loadCameraIntrinsic_KITTI_IMAGE0(const std::string& dir); // functions for loading yaml files.
};



#endif