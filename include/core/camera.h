#ifndef _CAMERA_H_
#define _CAMERA_H_

#include <iostream>
#include <vector>
#include <Eigen/Dense>

#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "core/image_processing.h"
#include "core/type_defines.h"


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

#endif