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

		int n_cols = img.cols;
		int n_rows = img.rows;

		float uc, vc; // float-precision coordinates
		int u0, v0; // truncated coordinates
		int idx_I1, idx_I2, idx_I3, idx_I4;

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
			
			u0 = floor(uc);
			v0 = floor(vc);

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

	void interpImageSameRatio(const cv::Mat& img, const PixelVec& pts,
		const float ax, const float ay, const float axay,
		std::vector<float>& interp_values, MaskVec& mask_valid)
	{
		const float* img_ptr = img.ptr<float>(0);
		int n_pts = pts.size();
		interp_values.resize(n_pts);
		mask_valid.resize(n_pts);

		int n_cols = img.cols;
		int n_rows = img.rows;

		int idx_I1, idx_I2, idx_I3, idx_I4;

		float I1, I2, I3, I4;
		// I1 ax / 1-ax I2 
		// ay   (v,u)
		//  
		// 1-ay
		// I3           I4

		for (uint32_t i = 0; i < n_pts; ++i) {
			bool is_valid = true;

			const float& uc = pts[i].x;
			const float& vc = pts[i].y;
			
			int u0 = (int)floor(uc);
			int v0 = (int)floor(vc);

			if (u0 < 1 && u0 >= n_cols-2) // inside the image
			{
				interp_values[i] = -2.0f;
				mask_valid[i]    = false;
				continue;
			}

			if (v0 < 1 && v0 >= n_rows-2)
			{
				interp_values[i] = -2.0f;
				mask_valid[i]    = false;
				continue;
			}

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

		int n_cols = img.cols;
		int n_rows = img.rows;

		float uc, vc; // float-precision coordinates
		int u0, v0; // truncated coordinates
		int idx_I1, idx_I2, idx_I3, idx_I4;

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
			
			u0 = floor(uc);
			v0 = floor(vc);

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

	float calcZNCC(const cv::Mat& img0, const cv::Mat& img1, const Pixel& pt0, const Pixel& pt1, int win_sz){
		int n_cols = img0.cols;
		int n_rows = img0.rows;

		if(img0.cols != img1.cols || img0.rows != img1.rows)
			throw std::runtime_error("in 'calcZNCC', img0.cols != img1.cols || img0.rows != img1.rows");
		
		if(!(win_sz & 0x01))
			throw std::runtime_error("in 'calcZNCC', !(win_sz & 0x01)");
			
		cv::Mat I0, I1;
		img0.convertTo(I0, CV_32FC1);
		img1.convertTo(I1, CV_32FC1);

		// Generate patch
		int win_half = win_sz/2;
		int n_elem = win_sz*win_sz;
		PixelVec patt(n_elem);
		int ind = 0;
		for(int i = 0; i < win_sz; ++i) {
			for(int j = 0; j < win_sz; ++j) {
				patt[ind].x = (float)(i-win_sz);
				patt[ind].y = (float)(j-win_sz);
				++ind;
			}
		}
		n_elem = ind;

		// interpolate patches
		std::vector<float> I0_patt(n_elem);
		std::vector<float> I1_patt(n_elem);		
		PixelVec patt0(n_elem);
		PixelVec patt1(n_elem);
		MaskVec mask0(n_elem);
		MaskVec mask1(n_elem);
		for(int j = 0; j < n_elem; ++j){
			patt0[j] = pt0 + patt[j];
			patt1[j] = pt1 + patt[j];
		}
        image_processing::interpImage(I0, patt0, I0_patt, mask0);
        image_processing::interpImage(I1, patt1, I1_patt, mask1);

		// calculate zncc

		float mean_l = 0;
		float mean_r = 0;
		float numer = 0;
		float denom_l = 0;
		float denom_r = 0;
		
		float cost_temp = -10;

		// calculate means
		for(int i = 0; i < n_elem; ++i){
			mean_l += I0_patt[i];
			mean_r += I1_patt[i];
		}
		mean_l /= (float)n_elem;
		mean_r /= (float)n_elem;

		// Calculate cost
		for(int i = 0; i < n_elem; ++i){
			float I0_ml = I0_patt[i] - mean_l;
			float I1_mr = I1_patt[i] - mean_r;
			numer += (I0_ml)*(I1_mr);
			denom_l += I0_ml*I0_ml;
			denom_r += I1_mr*I1_mr;
		}
		cost_temp = numer/sqrt(denom_l*denom_r);

		return cost_temp;
	};

	void interpImage3SameRatio(const cv::Mat& img, const cv::Mat& du, const cv::Mat& dv, const PixelVec& pts,
		const float ax, const float ay, const float axay,
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

		int n_cols = img.cols;
		int n_rows = img.rows;

		int idx_I1, idx_I2, idx_I3, idx_I4;

		float I1, I2, I3, I4;
		// I1 ax / 1-ax I2 
		// ay   (v,u)
		//  
		// 1-ay
		// I3           I4

		for (uint32_t i = 0; i < n_pts; ++i) {
			bool is_valid = true;

			const float& uc = pts[i].x;
			const float& vc = pts[i].y;
			
			int u0 = (int)floor(uc);
			int v0 = (int)floor(vc);

			if (u0 < 1 && u0 >= n_cols-2) // inside the image
			{
				interp_img[i] = -2.0f;
				interp_dv[i] = -2.0f;
				interp_dv[i] = -2.0f;
				mask_valid[i]    = false;
				continue;
			}

			if (v0 < 1 && v0 >= n_rows-2)
			{
				interp_img[i] = -2.0f;
				interp_dv[i] = -2.0f;
				interp_dv[i] = -2.0f;
				mask_valid[i]    = false;
				continue;
			}

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
	};

};