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


class Camera {
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

	const int cols() const { return n_cols_; };
	const int rows() const { return n_rows_; };
	const float fx() const { return fx_; };
	const float fy() const { return fy_; };
	const float cx() const { return cx_; };
	const float cy() const { return cy_; };
	const float fxinv() const { return fxinv_; };
	const float fyinv() const { return fyinv_; };
	const Eigen::Matrix3f K() const { return K_; };
	const Eigen::Matrix3f Kinv() const { return Kinv_; };
	const cv::Mat cvK() const { return cvK_; };

private:
	void generateImageUndistortMaps();
	void generatePixelUndistortMaps();
};

#endif