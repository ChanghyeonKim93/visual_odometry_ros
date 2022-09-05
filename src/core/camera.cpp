#include "core/camera.h"

Camera::Camera()
{
	// initialize all things
	K_ = Eigen::Matrix3f::Identity();
	Kinv_ = Eigen::Matrix3f::Identity();

	printf(" - CAMERA is constructed.\n");
};

Camera::~Camera() {
	printf(" - CAMERA is deleted.\n");
};

void Camera::initParams(int n_cols, int n_rows, const cv::Mat& cvK, const cv::Mat& cvD) {
	n_cols_ = n_cols; n_rows_ = n_rows;
	
	cvK.copyTo(cvK_);

	fx_ = cvK.at<float>(0, 0); fy_ = cvK.at<float>(1, 1);
	cx_ = cvK.at<float>(0, 2); cy_ = cvK.at<float>(1, 2);
	fxinv_ = 1.0f / fx_; fyinv_ = 1.0f / fy_;
	k1_ = cvD.at<float>(0, 0);
	k2_ = cvD.at<float>(0, 1);
	p1_ = cvD.at<float>(0, 2);
	p2_ = cvD.at<float>(0, 3);
	k3_ = cvD.at<float>(0, 4);
	distortion_[0] = k1_;
	distortion_[1] = k2_;
	distortion_[2] = p1_;
	distortion_[3] = p2_;
	distortion_[4] = k3_;

	K_ << fx_, 0.0f, cx_, 0.0f, fy_, cy_, 0.0f, 0.0f, 1.0f;
	Kinv_ = K_.inverse();

	distorted_map_u_ = cv::Mat::zeros(n_rows_, n_cols_, CV_32FC1);
	distorted_map_v_ = cv::Mat::zeros(n_rows_, n_cols_, CV_32FC1);
	
	undistorted_pixel_map_u_ = cv::Mat::zeros(n_rows_, n_cols_, CV_32FC1);
	undistorted_pixel_map_v_ = cv::Mat::zeros(n_rows_, n_cols_, CV_32FC1);
	
	this->generateImageUndistortMaps();
	this->generatePixelUndistortMaps();
	
	printf(" - CAMERA - 'initParams()' - : camera params incomes.\n");
};

void Camera::generateImageUndistortMaps() {
	float* map_x_ptr = nullptr;
	float* map_y_ptr = nullptr;
	float x, y, r2, r4, r6, r_radial, x_dist, y_dist, xy2, xx, yy;

	for (uint32_t v = 0; v < n_rows_; ++v) {
		map_x_ptr = distorted_map_u_.ptr<float>(v);
		map_y_ptr = distorted_map_v_.ptr<float>(v);
		y = ((float)v - cy_) * fyinv_;

		for (uint32_t u = 0; u < n_cols_; ++u) {
			x = ((float)u - cx_) * fxinv_;
			xy2 = 2.0 * x*y;
			xx = x * x; yy = y * y;
			r2 = xx + yy;
			r4 = r2 * r2;
			r6 = r4 * r2;
			// r = sqrt(r2);
			r_radial = 1.0 + k1_ * r2 + k2_ * r4 + k3_ * r6;
			x_dist = x * r_radial + p1_*xy2 + p2_ * (r2 + 2.0 * xx);
			y_dist = y * r_radial + p1_ * (r2 + 2.0 * yy) + p2_*xy2;

			*(map_x_ptr + u) = cx_ + x_dist * fx_;
			*(map_y_ptr + u) = cy_ + y_dist * fy_;
		}
	}
	printf(" - CAMERA - 'generateImageUndistortMaps()' ... \n");
};

void Camera::generatePixelUndistortMaps()
{
	float THRES_EPS = 1e-9;
	uint32_t MAX_ITER = 500;

	float* map_x_ptr = nullptr;
	float* map_y_ptr = nullptr;

	float xd, yd, R, R2, R3, D;
	float xdc, ydc, xy2, xx, yy;

	for(uint32_t v = 0; v < n_rows_; ++v){
		map_x_ptr = undistorted_pixel_map_u_.ptr<float>(v);
		map_y_ptr = undistorted_pixel_map_v_.ptr<float>(v);
		yd = ((float)v - cy_) * fyinv_;

		float err_prev = 1e12;
		for (uint32_t u = 0; u < n_cols_; ++u) {
			xd = ((float)u - cx_) * fxinv_;

			// Iteratively finding
			float x = xd;
			float y = yd;
			for(int iter = 0; iter < MAX_ITER; ++iter){
				xy2 = 2.0f * x * y;
				xx = x * x;
				yy = y * y;
				R  = xx + yy;
				R2 = R * R;
				R3 = R2 * R;

				D = 1.0 + k1_*R + k2_*R2 + k3_*R3;
				xdc = x*D + p1_*xy2 + p2_ * (R + 2.0f * xx);
				ydc = y*D + p1_ * (R + 2.0f * yy) + p2_*xy2;

				float dR_dx = 2.0f*x;
				float dR_dy = 2.0f*y;
				
				float dD_dx = (k1_ + 2.0f*k2_*R + 3.0f*k3_*R2)*dR_dx;
				float dD_dy = (k1_ + 2.0f*k2_*R + 3.0f*k3_*R2)*dR_dy;
				
				float dxdc_dx = D + x*dD_dx + 2.0f*p1_*y + p2_*dR_dx + 4.0f*p2_*x;
				float dxdc_dy =     x*dD_dy + 2.0f*p1_*x + p2_*dR_dy;
				
				float dydc_dx =     y*dD_dx + 2.0f*p2_*y + p1_*dR_dx;
				float dydc_dy = D + y*dD_dy + 2.0f*p2_*x + p1_*dR_dy + 4.0f*p1_*y;
				
				float rx = xdc - xd;
				float ry = ydc - yd;
				float dC_dxy[2];
				dC_dxy[0] = 2.0f*(rx*dxdc_dx + ry*dydc_dx);
				dC_dxy[1] = 2.0f*(rx*dxdc_dy + ry*dydc_dy);

				float err_curr = rx*rx + ry*ry;
				if(fabs(err_curr - err_prev) < THRES_EPS) break;

				// Update 
				x -= dC_dxy[0];
				y -= dC_dxy[1];

				err_prev = err_curr;
			}

			*(map_x_ptr + u) = cx_ + x * fx_;
			*(map_y_ptr + u) = cy_ + y * fy_;
		}
	}
	printf(" - CAMERA - 'generatePixelUndistortMaps()' ... \n");
};

void Camera::undistortImage(const cv::Mat& raw, cv::Mat& rectified) {
	if (raw.empty() || raw.cols != n_cols_ || raw.rows != n_rows_)
		throw std::runtime_error("undistort image: provided image has not the same size as the camera model!\n");

	cv::Mat img_float;
	if(raw.channels() == 3){
		std::cout << "Input image channels == 3\n";
		cv::cvtColor(raw, img_float, cv::COLOR_RGB2GRAY);
	}
	else raw.copyTo(img_float);

	if(img_float.type() != CV_32FC1 ){
		img_float.convertTo(img_float, CV_32FC1);
	}

	cv::remap(img_float, rectified, this->distorted_map_u_, this->distorted_map_v_, cv::INTER_LINEAR);
};

void Camera::undistortPixels(const PixelVec& pts_raw, PixelVec& pts_undist){ // 점만 undistort 하는 것. 연산량 매우 줄여줄 수 있다.
	uint32_t n_pts = pts_raw.size();
	pts_undist.resize(n_pts);

	MaskVec mask_valid;
	std::vector<float> u_undist;
	std::vector<float> v_undist;

	mask_valid.reserve(n_pts);
	u_undist.reserve(n_pts);
	v_undist.reserve(n_pts);

	image_processing::interpImage(undistorted_pixel_map_u_ , pts_raw, u_undist, mask_valid);
	image_processing::interpImage(undistorted_pixel_map_v_ , pts_raw, v_undist, mask_valid);

	for(int i = 0; i < n_pts; ++i){
		pts_undist[i].x = cx_ + u_undist[i] * fx_;
		pts_undist[i].y = cy_ + v_undist[i] * fy_;
	}
};

Pixel Camera::projectToPixel(const Point& X){
	float invz = 1.0f/X(2);

	return Pixel(fx_*X(0)*invz+cx_,fy_*X(1)*invz+cy_);
};

Point Camera::reprojectToNormalizedPoint(const Pixel& pt){
	return Point(fxinv_*(pt.x-cx_), fyinv_*(pt.y-cy_), 1.0f);
};