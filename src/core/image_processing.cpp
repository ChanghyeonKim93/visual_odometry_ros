#include "core/image_processing.h"

namespace image_processing {
	std::string type2str(cv::Mat img) {
		std::string r;
		int type = img.type();

		uchar depth = type & CV_MAT_DEPTH_MASK;
		uchar chans = 1 + (type >> CV_CN_SHIFT);

		switch (depth) {
			case CV_8U:  r = "8U";   break;
			case CV_8S:  r = "8S";   break;
			case CV_16U: r = "16U";  break;
			case CV_16S: r = "16S";  break;
			case CV_32S: r = "32S";  break;
			case CV_32F: r = "32F";  break;
			case CV_64F: r = "64F";  break;
			default:     r = "User"; break;
		}

		r += "C";
		r += (chans + '0');

		return r;
	};

	void interpImage(const cv::Mat& img, const PixelVec& pts,
		std::vector<float>& interp_values, MaskVec& mask_valid)
	{
		const float* img_ptr = img.ptr<float>(0);
		uint32_t n_pts = pts.size();
		interp_values.resize(n_pts);
		mask_valid.resize(n_pts);

		uint32_t n_cols = img.cols;
		uint32_t n_rows = img.rows;

		float uc, vc; // float-precision coordinates
		uint32_t u0, v0; // truncated coordinates
		uint32_t idx_I1, idx_I2, idx_I3, idx_I4;

		float I1, I2, I3, I4;
		float ax, ay, axay;

		// I1 ax / 1-ax I2 
		// ay   (v,u)
		//  
		// 1-ay
		// I3           I4

		for (uint32_t i = 0; i < n_pts; ++i) {
			bool is_valid = true;

			uc = pts[i].x;
			vc = pts[i].y;
			
			u0 = (uint32_t)floor(uc);
			v0 = (uint32_t)floor(vc);

			if (u0 >= 0 && u0 < n_cols-1) // inside the image
				ax = uc - (float)u0; // should be 0 <= ax <= 1
			else {
				interp_values[i] = -2.0f;
				mask_valid[i]    = false;
				continue;
			}

			if (v0 >= 0 && v0 < n_rows - 1)
				ay = vc - (float)v0;
			else {
				interp_values[i] = -2.0f;
				mask_valid[i]    = false;
				continue;
			}

			axay   = ax*ay;
			idx_I1 = v0*n_cols + u0;

			const float* p = img_ptr + idx_I1;
			I1 = *p;             // v_0n_colsu_0
			I2 = *(++p);         // v_0n_colsu_0 + 1
			I4 = *(p += n_cols); // v_0n_colsu_0 + 1 + n_cols
			I3 = *(--p);      // v_0n_colsu_0 + n_cols

			interp_values[i] = axay*(I1 - I2 - I3 + I4) + ax*(-I1 + I2) + ay*(-I1 + I3) + I1;
			mask_valid[i]    = true;
			
		}	
	};

	void interpImage3(const cv::Mat& img, const cv::Mat& du, const cv::Mat& dv, const PixelVec& pts,
		std::vector<float>& interp_img, std::vector<float>& interp_du, std::vector<float>& interp_dv, MaskVec& mask_valid)
	{
		const float* img_ptr = img.ptr<float>(0);
		const float* du_ptr = du.ptr<float>(0);
		const float* dv_ptr = dv.ptr<float>(0);
		uint32_t n_pts = pts.size();
		interp_img.resize(n_pts);
		interp_du.resize(n_pts);
		interp_dv.resize(n_pts);
		mask_valid.resize(n_pts);

		uint32_t n_cols = img.cols;
		uint32_t n_rows = img.rows;

		float uc, vc; // float-precision coordinates
		uint32_t u0, v0; // truncated coordinates
		uint32_t idx_I1, idx_I2, idx_I3, idx_I4;

		float I1, I2, I3, I4;
		float ax, ay, axay;

		// I1 ax / 1-ax I2 
		// ay   (v,u)
		//  
		// 1-ay
		// I3           I4

		for (uint32_t i = 0; i < n_pts; ++i) {
			bool is_valid = true;

			uc = pts[i].x;
			vc = pts[i].y;
			
			u0 = (uint32_t)floor(uc);
			v0 = (uint32_t)floor(vc);

			if (u0 >= 0 && u0 < n_cols-1) // inside the image
				ax = uc - (float)u0; // should be 0 <= ax <= 1
			else {
				interp_img[i] = -2.0f;
				interp_dv[i] = -2.0f;
				interp_dv[i] = -2.0f;
				mask_valid[i]    = false;
				continue;
			}

			if (v0 >= 0 && v0 < n_rows - 1)
				ay = vc - (float)v0;
			else {
				interp_img[i] = -2.0f;
				interp_dv[i] = -2.0f;
				interp_dv[i] = -2.0f;
				mask_valid[i]    = false;
				continue;
			}

			axay   = ax*ay;
			idx_I1 = v0*n_cols + u0;

			const float* p = img_ptr + idx_I1;
			I1 = *p;             // v_0n_colsu_0
			I2 = *(++p);         // v_0n_colsu_0 + 1
			I4 = *(p += n_cols); // v_0n_colsu_0 + 1 + n_cols
			I3 = *(--p);      // v_0n_colsu_0 + n_cols

			interp_img[i] = axay*(I1 - I2 - I3 + I4) + ax*(-I1 + I2) + ay*(-I1 + I3) + I1;

			p = du_ptr + idx_I1;
			I1 = *p;             // v_0n_colsu_0
			I2 = *(++p);         // v_0n_colsu_0 + 1
			I4 = *(p += n_cols); // v_0n_colsu_0 + 1 + n_cols
			I3 = *(--p);      // v_0n_colsu_0 + n_cols

			interp_du[i] = axay*(I1 - I2 - I3 + I4) + ax*(-I1 + I2) + ay*(-I1 + I3) + I1;

			p = dv_ptr + idx_I1;
			I1 = *p;             // v_0n_colsu_0
			I2 = *(++p);         // v_0n_colsu_0 + 1
			I4 = *(p += n_cols); // v_0n_colsu_0 + 1 + n_cols
			I3 = *(--p);      // v_0n_colsu_0 + n_cols

			interp_dv[i] = axay*(I1 - I2 - I3 + I4) + ax*(-I1 + I2) + ay*(-I1 + I3) + I1;
			
			mask_valid[i]    = true;
		}	
	}
};