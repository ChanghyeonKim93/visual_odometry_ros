#ifndef _CAMERA_H_
#define _CAMERA_H_

#include <iostream>
#include <memory>
#include <vector>

#include "eigen3/Eigen/Dense"

#include "opencv4/opencv2/core.hpp"
#include "opencv4/opencv2/imgproc.hpp"
#include "opencv4/opencv2/highgui.hpp"
#include "opencv4/opencv2/core/eigen.hpp"

#include "core/defines/define_type.h"

#include "core/util/image_processing.h"

/// @brief Camera class.
class Camera 
{
private:
	int n_cols_, n_rows_;

	Eigen::Matrix3f K_;
	Eigen::Matrix3f Kinv_;

	cv::Mat cvK_;

	float fx_, fy_, cx_, cy_;
	float fxinv_, fyinv_;
	float distortion_[5];
	float k1_, k2_, k3_, p1_, p2_;

	// image undistortion maps
	cv::Mat distorted_map_u_; // CV_32FC1
	cv::Mat distorted_map_v_; // CV_32FC1

	// Pixel undistortion maps
	cv::Mat undistorted_pixel_map_u_; //CV_32FC1
	cv::Mat undistorted_pixel_map_v_; //CV_32FC1

public:
	/// @brief Camera class constructor
	Camera();

	/// @brief Camera class destructor
	~Camera();

	/// @brief Initialize parameter of camera
	/// @param n_cols image columns
	/// @param n_rows image rows
	/// @param cvK intrinsic matrix in cv::Mat (3 x 3)
	/// @param cvD distortion parameters in cv::Mat (5 x 1, k1,k2,p1,p2,k3)
	void initParams(int n_cols, int n_rows, const cv::Mat& cvK, const cv::Mat& cvD);

	/// @brief Undistort the raw image
	/// @param raw distorted raw image
	/// @param rectified undistorted image (CV_32FC1)
	void undistortImage(const cv::Mat& raw, cv::Mat& rectified);


	/// @brief Undistort pixels. Undistorted pixel positions are interpolated based on the undistortion map (pre-built)
	/// @param pts_raw distorted pixels (extracted on the distorted raw image)
	/// @param pts_undist undistorted pixels 
	void undistortPixels(const PixelVec& pts_raw, PixelVec& pts_undist);

	/// @brief Project 3D point onto 2D pixel plane
	/// @param X 3D point represented in the frame where 3D point is going to be projected
	/// @return 2D projected pixel
	Pixel projectToPixel(const Point& X);

	/// @brief Reproject 2D pixel point to the 3D point
	/// @param pt 2D pixel point
	/// @return 3D point reprojected from the 2D pixel
	Point reprojectToNormalizedPoint(const Pixel& pt);

	bool inImage(const Pixel& pt);
	bool inImage(Pixel& pt);

	/// @brief Get # of image pixel columns of this camera.
	/// @return # of image pixel columns 
	const int cols() const { return n_cols_; };

	/// @brief Get # of image pixel rows of this camera.
	/// @return # of image pixel rows 
	const int rows() const { return n_rows_; };

	/// @brief Get focal length of this camera. x-axis (== u-axis)
	/// @return focal length x-axis ( == u-axis)
	const float fx() const { return fx_; };

	/// @brief Get focal length of this camera. y-axis (== v-axis)
	/// @return focal length y-axis ( == v-axis)
	const float fy() const { return fy_; };

	/// @brief Get image pixel center along x-axis (== u-axis)
	/// @return image pixel center long x-axis (== u-axis)
	const float cx() const { return cx_; };

	/// @brief Get image pixel center along y-axis (== v-axis)
	/// @return image pixel center long y-axis (== v-axis)
	const float cy() const { return cy_; };

	/// @brief Get inverse of focal length x
	/// @return inverse of focal length x (1.0/fx)
	const float fxinv() const { return fxinv_; };

	/// @brief Get inverse of focal length y
	/// @return inverse of focal length y (1.0/fy)
	const float fyinv() const { return fyinv_; };

	const float k1() const { return k1_; };
	const float k2() const { return k2_; };
	const float k3() const { return k3_; };
	const float p1() const { return p1_; };
	const float p2() const { return p2_; };

	/// @brief Get 3x3 intrinsic matrix of this camera
	/// @return 3x3 intrinsic matrix (in Eigen Matrix)
	const Eigen::Matrix3f K() const { return K_; };

	/// @brief Get inverse of 3x3 intrinsic matrix of this camera
	/// @return inverse of 3x3 intrinsic matrix (in Eigen Matrix)
	const Eigen::Matrix3f Kinv() const { return Kinv_; };

	/// @brief Get 3x3 intrinsic matrix of this camera
	/// @return 3x3 intrinsic matrix (in OpenCV cv::Mat format)
	const cv::Mat cvK() const { return cvK_; };

private:
	/// @brief Generate pre-calculated image undistortion maps 
	void generateImageUndistortMaps();

	/// @brief Generate pre-calculated pixel undistortion maps (It can be considered as inverse mapping of image undistortion maps.)
	void generatePixelUndistortMaps();
};


class StereoCamera
{
private:
	CameraPtr cam_left_;
	CameraPtr cam_right_;

	PoseSE3 T_lr_;
	PoseSE3 T_rl_;

private:
	CameraPtr cam_rect_; // Rectified camera
	PoseSE3 T_lr_rect_; // Rectified left to right pose (Rotation: Identity)
	PoseSE3 T_rl_rect_; // Rectified right to left pose (Rotation: Identity)

	// stereo rectification maps
	cv::Mat rectify_map_left_u_; // CV_32FC1
	cv::Mat rectify_map_left_v_; // CV_32FC1
	cv::Mat rectify_map_right_u_; // CV_32FC1
	cv::Mat rectify_map_right_v_; // CV_32FC1

	bool is_initialized_to_stereo_rectify_;

public:
	StereoCamera();
	~StereoCamera();

	void setStereoPoseLeft2Right(const PoseSE3& T_lr);

	void initStereoCameraToRectify();

public:
	void undistortImageByLeftCamera(const cv::Mat& img_left, cv::Mat& img_left_undist);
	void undistortImageByRightCamera(const cv::Mat& img_right, cv::Mat& img_right_undist);
	void rectifyStereoImages(
		const cv::Mat& img_left, const cv::Mat& img_right,
		cv::Mat& img_left_rect, cv::Mat& img_right_rect); // todo

// Get methods
public:
	CameraConstPtr& getLeftCamera() const; 
	CameraConstPtr& getRightCamera() const; 

	const PoseSE3& getStereoPoseLeft2Right() const;
	const PoseSE3& getStereoPoseRight2Left() const;

public:
	CameraConstPtr& getRectifiedCamera() const; 
	const PoseSE3& getRectifiedStereoPoseLeft2Right() const;
	const PoseSE3& getRectifiedStereoPoseRight2Left() const;


private:
	void generateStereoImagesUndistortAndRectifyMaps();
};

#endif