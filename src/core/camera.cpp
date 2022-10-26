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


bool Camera::inImage(const Pixel& pt)
{
	bool is_in_image = true;
	float offset = 3.0f;

	if(pt.x < offset || pt.y < offset || pt.x >= n_cols_-offset || pt.y >= n_rows_-offset)
		is_in_image = false;

	return is_in_image;
};

bool Camera::inImage(Pixel& pt)
{
	bool is_in_image = true;
	float offset = 3.0f;

	if(pt.x < offset || pt.y < offset || pt.x >= n_cols_-offset || pt.y >= n_rows_-offset)
		is_in_image = false;

	return is_in_image;
};


/*
====================================================================================
====================================================================================
=============================  StereoCamera  =============================
====================================================================================
====================================================================================
*/
StereoCamera::StereoCamera()
: cam_left_(nullptr), cam_right_(nullptr), 
is_initialized_to_stereo_rectify_(false)
{
	cam_left_  = std::make_shared<Camera>();
	cam_right_ = std::make_shared<Camera>();
};

StereoCamera::~StereoCamera()
{

};

CameraConstPtr& StereoCamera::getLeftCamera() const
{
	return cam_left_;
};

CameraConstPtr& StereoCamera::getRightCamera() const
{
	return cam_right_;
};

CameraConstPtr& StereoCamera::getRectifiedCamera() const
{
	if( !is_initialized_to_stereo_rectify_ ) throw std::runtime_error("In 'getRectifiedCamera()', is_initialized_to_stereo_rectify_ == false");
	return cam_rect_;
};

void StereoCamera::setStereoPoseLeft2Right(const PoseSE3& T_lr)
{
	T_lr_ = T_lr;
	T_rl_ = T_lr_.inverse();
};

void StereoCamera::initStereoCameraToRectify()
{
	this->generateStereoImagesUndistortAndRectifyMaps();
	is_initialized_to_stereo_rectify_ = true;
}

void StereoCamera::undistortImageByLeftCamera(const cv::Mat& img_left, cv::Mat& img_left_undist)
{
	cam_left_->undistortImage(img_left,img_left_undist);
};

void StereoCamera::undistortImageByRightCamera(const cv::Mat& img_right, cv::Mat& img_right_undist)
{
	cam_right_->undistortImage(img_right,img_right_undist);
};

void StereoCamera::rectifyStereoImages(
	const cv::Mat& img_left, const cv::Mat& img_right,
	cv::Mat& img_left_rect, cv::Mat& img_right_rect)
{
	if( !is_initialized_to_stereo_rectify_ ) throw std::runtime_error("In 'rectifyStereoImages()', is_initialized_to_stereo_rectify_ == false");

	// LEFT IMAGE
	if (img_left.empty() || img_left.cols != cam_rect_->cols() || img_left.rows != cam_rect_->rows())
		throw std::runtime_error("In 'rectifyStereoImages()': provided image has not the same size as the camera model!\n");

	cv::Mat img_left_float;
	if(img_left.channels() == 3)
		cv::cvtColor(img_left, img_left_float, cv::COLOR_RGB2GRAY);
	else img_left.copyTo(img_left_float);

	if(img_left_float.type() != CV_32FC1 )
		img_left_float.convertTo(img_left_float, CV_32FC1);

	cv::remap(img_left_float, img_left_rect, rectify_map_left_u_, rectify_map_left_v_, cv::INTER_LINEAR);
	
	// RIGHT IMAGE
	if (img_right.empty() || img_right.cols != cam_rect_->cols() || img_right.rows != cam_rect_->rows())
		throw std::runtime_error("In 'rectifyStereoImages()': provided image has not the same size as the camera model!\n");
	
	cv::Mat img_right_float;
	if(img_right.channels() == 3)
		cv::cvtColor(img_right, img_right_float, cv::COLOR_RGB2GRAY);
	else img_right.copyTo(img_right_float);

	if(img_right_float.type() != CV_32FC1 )
		img_right_float.convertTo(img_right_float, CV_32FC1);

	cv::remap(img_right_float, img_right_rect, rectify_map_right_u_, rectify_map_right_v_, cv::INTER_LINEAR);
};

const PoseSE3& StereoCamera::getStereoPoseLeft2Right() const
{
	return T_lr_;
};

const PoseSE3& StereoCamera::getStereoPoseRight2Left() const
{
	return T_rl_;
};


const PoseSE3& StereoCamera::getRectifiedStereoPoseLeft2Right() const
{	
	if( !is_initialized_to_stereo_rectify_ ) throw std::runtime_error("In 'getRectifiedStereoPoseLeft2Right()', is_initialized_to_stereo_rectify_ == false");

	return T_lr_rect_;	
};

const PoseSE3& StereoCamera::getRectifiedStereoPoseRight2Left() const
{
	if( !is_initialized_to_stereo_rectify_ ) throw std::runtime_error("In 'getRectifiedStereoPoseRight2Left()', is_initialized_to_stereo_rectify_ == false");

	return T_rl_rect_;	
};


void StereoCamera::generateStereoImagesUndistortAndRectifyMaps()
{
	float scale = 2.0f;
	float invscale = 1.0f/scale;

    PoseSE3 T_0l = PoseSE3::Identity();
    PoseSE3 T_0r = T_lr_;

    const Rot3& R_0l = T_0l.block<3, 3>(0, 0);
    const Rot3& R_0r = T_0r.block<3, 3>(0, 0);
	const Pos3& t_0r = T_0r.block<3, 1>(0, 3);

    Rot3 R_l0 = R_0l.transpose();
    Rot3 R_r0 = R_0r.transpose();

	// Generate Reference rotation matrix
    Vec3 k_l = R_0l.block<3, 1>(0, 2);
    Vec3 k_r = R_0r.block<3, 1>(0, 2);
    Vec3 k_n = (k_l + k_r)*0.5;
    k_n /= k_n.norm();
	
	int n_cols = cam_left_->cols();
	int n_rows = cam_left_->rows();
	
	Vec3 i_n = t_0r;
    i_n /= i_n.norm();

    Vec3 j_n = k_n.cross(i_n);
    j_n /= j_n.norm();

	k_n = i_n.cross(j_n);
	k_n /= k_n.norm();

	std::cout << "i dot j: " << i_n.dot(j_n) << ", "
			  << "j dot k: " << j_n.dot(k_n) << ", "
			  << "k dot i: " << k_n.dot(i_n) << "\n";

    Rot3 R_0n; // left to rectified camera
	R_0n << i_n, j_n, k_n;

	// New intrinsic parameter
    float f_n = 
		(cam_left_->fx() + cam_left_->fy()) * invscale;

    float centu = (float)cam_left_->cols()*0.5f;
    float centv = (float)cam_left_->rows()*0.5f;
    
	Mat33 K_rect;
    K_rect << f_n,   0, centu, 
			    0, f_n, centv,
				0,   0,     1;

	Mat33 K_rect_inv;
    K_rect_inv = K_rect.inverse();

	Eigen::Matrix<float,1,5> D_rect;
	D_rect << 0,0,0,0,0;

	cv::Mat cvK_rect;
	cv::Mat cvD_rect;
    cv::eigen2cv(K_rect, cvK_rect);
    cv::eigen2cv(D_rect, cvD_rect);

	// Generate rectified camera
	cam_rect_ = std::make_shared<Camera>();
	cam_rect_->initParams(n_cols, n_rows, cvK_rect, cvD_rect);

	rectify_map_left_u_  = cv::Mat::zeros(n_rows, n_cols, CV_32FC1);
	rectify_map_left_v_  = cv::Mat::zeros(n_rows, n_cols, CV_32FC1);
	rectify_map_right_u_ = cv::Mat::zeros(n_rows, n_cols, CV_32FC1);
	rectify_map_right_v_ = cv::Mat::zeros(n_rows, n_cols, CV_32FC1);

    // interpolation grid calculations.

	Vec3 p_n;
    Vec3 P_0, x_l, x_r;

    float k1, k2, k3, p1, p2;
    float x, y, xy, r2, r4, r6, r_radial, x_dist, y_dist;

	const float& fx_l = cam_left_->fx();
	const float& fy_l = cam_left_->fy();
	const float& cx_l = cam_left_->cx();
	const float& cy_l = cam_left_->cy();

	const float& fx_r = cam_right_->fx();
	const float& fy_r = cam_right_->fy();
	const float& cx_r = cam_right_->cx();
	const float& cy_r = cam_right_->cy();

    float* ptr_map_left_u = nullptr;
	float* ptr_map_left_v = nullptr;
    float* ptr_map_right_u = nullptr;
	float* ptr_map_right_v = nullptr;
    for (int v = 0; v < n_rows; ++v)
    {
        ptr_map_left_u = this->rectify_map_left_u_.ptr<float>(v);
        ptr_map_left_v = this->rectify_map_left_v_.ptr<float>(v);

        ptr_map_right_u = this->rectify_map_right_u_.ptr<float>(v);
        ptr_map_right_v = this->rectify_map_right_v_.ptr<float>(v);

        for (int u = 0; u < n_cols; ++u)
        {
			p_n << (float)u, (float)v, 1.0f;
            P_0 = R_0n*K_rect_inv*p_n;

            x_l = R_l0*P_0;
            x_l /= x_l(2); // left normalized coordinate

            x_r = R_r0*P_0;
            x_r /= x_r(2);// right normalized coordinate



		// Left
            k1 = cam_left_->k1();
            k2 = cam_left_->k2();
            k3 = cam_left_->k3();
            p1 = cam_left_->p1();
            p2 = cam_left_->p2();

            x = x_l(0);
            y = x_l(1);

			xy = x*y;
            r2 = x*x + y*y;
            r4 = r2*r2;
            r6 = r4*r2;

            r_radial = 1.0f + k1*r2 + k2*r4 + k3*r6;
            // x_dist = x*r_radial + 2.0f * p1*xy + p2*(r2 + 2.0f * x*x);
            // y_dist = y*r_radial + p1*(r2 + 2.0f * y*y) + 2.0f * p2*xy;
			x_dist = x*r_radial;
            y_dist = y*r_radial;

            *(ptr_map_left_u + u) = cx_l  +  x_dist * fx_l;
            *(ptr_map_left_v + u) = cy_l  +  y_dist * fy_l;



		// Right
            k1 = cam_right_->k1();
            k2 = cam_right_->k2();
            k3 = cam_right_->k3();
            p1 = cam_right_->p1();
            p2 = cam_right_->p2();

            x = x_r(0);
            y = x_r(1);

			xy = x*y;
            r2 = x*x + y*y;
            r4 = r2*r2;
            r6 = r4*r2;

            r_radial = 1.0f + k1*r2 + k2*r4 + k3*r6;
            // x_dist = x*r_radial + 2.0f * p1*xy + p2*(r2 + 2.0f * x*x);
            // y_dist = y*r_radial + p1*(r2 + 2.0f * y*y) + 2.0f * p2*xy;
			x_dist = x*r_radial;
            y_dist = y*r_radial;
            *(ptr_map_right_u + u) = cx_r  +  x_dist * fx_r;
            *(ptr_map_right_v + u) = cy_r  +  y_dist * fy_r;
        }
    }

    // Rectified stereo extrinsic parameters (T_nlnr)
    Rot3 R_ln = R_l0*R_0n;
    Rot3 R_rn = R_r0*R_0n;
    Pos3 t_clcr = T_lr_.block<3, 1>(0, 3);
    this->T_lr_rect_ << Rot3::Identity(), R_ln.transpose()*t_clcr, 0, 0, 0, 1;
    this->T_rl_rect_ << Rot3::Identity(), -R_ln.transpose()*t_clcr, 0, 0, 0, 1;

	std::cout << "T_lr_rect_:\n" << T_lr_rect_ << std::endl;
	
	is_initialized_to_stereo_rectify_ = true;
    std::cout << "[** INFO **] StereoCamera: stereo rectification maps are generated.\n";
};
