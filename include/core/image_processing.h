#ifndef _IMAGE_PROCESSING_H_
#define _IMAGE_PROCESSING_H_
#include <iostream>
#include <string>
#include <opencv2/core.hpp>
#include "core/type_defines.h"

namespace image_processing {
	std::string type2str(cv::Mat img);
	void interpImage(const cv::Mat& img, const PixelVec& pts,
		std::vector<float>& interp_values, MaskVec& mask_valid);
	void interpImage3(const cv::Mat& img, const cv::Mat& du, const cv::Mat& dv, const PixelVec& pts,
		std::vector<float>& interp_img, std::vector<float>& interp_du, std::vector<float>& interp_dv, MaskVec& mask_valid);
	
	float calcZNCC(const cv::Mat& img0, const cv::Mat& img1, const Pixel& pt0, const Pixel& pt1, int win_sz);
	
};
#endif