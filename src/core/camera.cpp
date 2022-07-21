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

	undist_map_x_ = cv::Mat::zeros(n_rows_, n_cols_, CV_32FC1);
	undist_map_y_ = cv::Mat::zeros(n_rows_, n_cols_, CV_32FC1);

	this->generateUndistortMaps();
	printf(" - CAMERA - 'initParams()' - : camera params incomes.\n");
};

void Camera::generateUndistortMaps() {
	float* map_x_ptr = nullptr;
	float* map_y_ptr = nullptr;
	double x, y, r, r2, r4, r6, r_radial, x_dist, y_dist, xy2, xx, yy;

	for (int v = 0; v < n_rows_; ++v) {
		map_x_ptr = undist_map_x_.ptr<float>(v);
		map_y_ptr = undist_map_y_.ptr<float>(v);
		y = ((double)v - (double)cy_) * (double)fyinv_;

		for (int u = 0; u < n_cols_; ++u) {
			x = ((double)u - (double)cx_) * (double)fxinv_;
			xy2 = 2.0 * x*y;
			xx = x * x; yy = y * y;
			r2 = xx + yy;
			r4 = r2 * r2;
			r6 = r4 * r2;
			// r = sqrt(r2);
			r_radial = 1.0 + k1_ * r2 + k2_ * r4 + k3_ * r6;
			x_dist = x * r_radial + p1_*xy2 + p2_ * (r2 + 2.0 * xx);
			y_dist = y * r_radial + p1_ * (r2 + 2.0 * yy) + p2_*xy2;

			*(map_x_ptr + u) = (double)cx_ + x_dist * (double)fx_;
			*(map_y_ptr + u) = (double)cy_ + y_dist * (double)fy_;
		}
	}
	printf(" - CAMERA - 'generateUndistortMaps()' ... \n");
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

	cv::remap(img_float, rectified, this->undist_map_x_, this->undist_map_y_, cv::INTER_LINEAR);
};

void Camera::undistortPixels(const PixelVec& pts_raw, PixelVec& pts_undist){
	uint32_t n_pts = pts_raw.size();
	pts_undist.resize(n_pts);


	float x, y, r, r2, r4, r6, r_radial, x_dist, y_dist, xy2, xx, yy;

	for(int i = 0; i < n_pts; ++i){
		x = fxinv_*(pts_raw[i].x-cx_);
		y = fyinv_*(pts_raw[i].y-cy_);
				
		xy2 = 2.0 * x*y;
		xx = x * x; yy = y * y;
		r2 = xx + yy;
		r4 = r2 * r2;
		r6 = r4 * r2;
		// r = sqrt(r2);
		r_radial = 1.0 + k1_ * r2 + k2_ * r4 + k3_ * r6;
		x_dist = x * r_radial + p1_*xy2 + p2_ * (r2 + 2.0 * xx);
		y_dist = y * r_radial + p1_ * (r2 + 2.0 * yy) + p2_*xy2;

		pts_undist[i].x = cx_ + x_dist * fx_;
		pts_undist[i].y = cy_ + y_dist * fy_;
	}
};
