#ifndef _SCALE_MONO_VO_H_
#define _SCALE_MONO_VO_H_

#include <iostream>
#include <exception>
#include <string>

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
#include "defines.h"

// custom
#include "camera.h"
#include "feature_extractor.h"

#include "image_processing.h"
#include "dataset_loader.h"

#include "timer.h"

class ScaleMonoVO
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
	std::shared_ptr<Camera> cam_;

// Feature related
private:
	std::shared_ptr<FeatureExtractor> extractor_;
	//std::shared_ptr<FeatureTracker>   tracker_;
	//std::shared_ptr<DataBase>         database_;
	//std::shared_ptr<MotionTracker>    motion_tracker_;

// dataset related.
private:
	dataset_loader::DatasetStruct dataset;

public:
	ScaleMonoVO(std::string mode);
	~ScaleMonoVO();

	void trackMonocular(const cv::Mat& img, const double& timestamp);

private:
	void runDataset(); // run algorithm

private:
	void loadCameraIntrinsic_KITTI_IMAGE0(const std::string& dir); // functions for loading yaml files.
};



#endif