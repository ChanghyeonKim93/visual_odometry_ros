#include "wrapper/ros1/stereo_rectifier/stereo_rectifier.h"

StereoRectifier::StereoRectifier(ros::NodeHandle& nh)
:stereo_cam_(nullptr), nh_(nh), directory_intrinsic_("")
{
    if(!ros::param::has("~directory_intrinsic"))
        throw std::runtime_error("'~directory_intrinsic' is not set.");
    ros::param::get("~directory_intrinsic", directory_intrinsic_);
    
    if(!ros::param::has("~topicname_image_left"))
        throw std::runtime_error("'~topicname_image_left' is not set.");
    ros::param::get("~topicname_image_left", topicname_image_left_);

    if(!ros::param::has("~topicname_image_right"))
        throw std::runtime_error("'~topicname_image_right' is not set.");
    ros::param::get("~topicname_image_right", topicname_image_right_);

    
    if(!ros::param::has("~topicname_image_left_rect"))
        throw std::runtime_error("'~topicname_image_left_rect' is not set.");
    ros::param::get("~topicname_image_left_rect", topicname_image_left_rect_);

    if(!ros::param::has("~topicname_image_right_rect"))
        throw std::runtime_error("'~topicname_image_right_rect' is not set.");
    ros::param::get("~topicname_image_right_rect", topicname_image_right_rect_);

    // Subscriber    
	left_img_sub_  = new message_filters::Subscriber<sensor_msgs::Image>(
        nh_, topicname_image_left_, 1);
	right_img_sub_ = new message_filters::Subscriber<sensor_msgs::Image>(
        nh_, topicname_image_right_, 1);
	sync_stereo_   = new message_filters::Synchronizer<MySyncPolicy>(
        MySyncPolicy(2), *left_img_sub_, *right_img_sub_);
    sync_stereo_->registerCallback(boost::bind(&StereoRectifier::imageStereoCallback, this, _1, _2));  
    
    // Publisher
    pub_left_rect_  = nh_.advertise<sensor_msgs::Image>(topicname_image_left_rect_,1);
    pub_right_rect_ = nh_.advertise<sensor_msgs::Image>(topicname_image_right_rect_,1);

	// Initialize stereo camera
	stereo_cam_ = std::make_shared<StereoCamera>();
	
    // load stereo parameters
    this->loadStereoCameraIntrinsics(directory_intrinsic_);

	// Get rectified stereo camera parameters
	CameraConstPtr& cam_rect = stereo_cam_->getRectifiedCamera();
	const PoseSE3& T_lr      = stereo_cam_->getRectifiedStereoPoseLeft2Right(); // left to right pose (rectified camera)

    // Run!
    this->run();
};


void StereoRectifier::imageStereoCallback(
    const sensor_msgs::ImageConstPtr &msg_left, const sensor_msgs::ImageConstPtr &msg_right)
{
    ros::Time t_callback_start = ros::Time::now();

    cv_bridge::CvImageConstPtr cv_left_ptr, cv_right_ptr;
    try {
        cv_left_ptr  = cv_bridge::toCvShare(msg_left);
        cv_right_ptr = cv_bridge::toCvShare(msg_right);
    } 
    catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        throw std::runtime_error("ERR!!!!");
        return;
    }    
    
    // Do stereo rectification.
    ros::Time t_track_start = ros::Time::now();
    double time_now = cv_left_ptr->header.stamp.toSec();

	cv::Mat img_left_undist, img_right_undist;
    stereo_cam_->rectifyStereoImages(
        cv_left_ptr->image, cv_right_ptr->image, 
        img_left_undist, img_right_undist);

    img_left_undist.convertTo(img_left_undist, CV_8UC1);
    img_right_undist.convertTo(img_right_undist, CV_8UC1);

    // publish
    cv_bridge::CvImage msg_crop;
    msg_crop.header.stamp    = ros::Time::now(); // Same times
    msg_crop.encoding        = "mono8"; // Or whatever
    msg_crop.image           = img_left_undist; // Your cv::Mat
    pub_left_rect_.publish(msg_crop);

    msg_crop.header.stamp    = ros::Time::now(); // Same times
    msg_crop.encoding        = "mono8"; // Or whatever
    msg_crop.image           = img_right_undist; // Your cv::Mat
    pub_right_rect_.publish(msg_crop);

    ros::Time t_track_end = ros::Time::now();

    ROS_INFO_STREAM("Stereo images income. Rectification time: " << (t_track_end-t_track_start).toNSec()*0.0000001 << " [ms]");
};

void StereoRectifier::run()
{
    ros::Rate rate(1000);
    while(ros::ok()){
        ros::spinOnce();
        rate.sleep();
    }
};

void StereoRectifier::loadStereoCameraIntrinsics(const std::string& dir)
{
    cv::FileStorage fs(dir, cv::FileStorage::READ);
	if (!fs.isOpened()) throw std::runtime_error("stereo intrinsic file cannot be found!\n");

// Left camera
	int rows, cols;
	rows = fs["Camera.left.height"];	cols = fs["Camera.left.width"];

	float fx, fy, cx, cy;
	fx = fs["Camera.left.fx"];	fy = fs["Camera.left.fy"];
	cx = fs["Camera.left.cx"];	cy = fs["Camera.left.cy"];

	float k1,k2,k3,p1,p2;
	k1 = fs["Camera.left.k1"];	k2 = fs["Camera.left.k2"];	k3 = fs["Camera.left.k3"];
	p1 = fs["Camera.left.p1"];	p2 = fs["Camera.left.p2"];

	cv::Mat cvK_tmp;
	cvK_tmp = cv::Mat(3,3,CV_32FC1);
	cvK_tmp.at<float>(0,0) = fx;	cvK_tmp.at<float>(0,1) = 0.0f;	cvK_tmp.at<float>(0,2) = cx;
	cvK_tmp.at<float>(1,0) = 0.0f;	cvK_tmp.at<float>(1,1) = fy;	cvK_tmp.at<float>(1,2) = cy;
	cvK_tmp.at<float>(2,0) = 0.0f;	cvK_tmp.at<float>(2,1) = 0.0f;	cvK_tmp.at<float>(2,2) = 1.0f;
	
	cv::Mat cvD_tmp;
	cvD_tmp = cv::Mat(1,5,CV_32FC1);
	cvD_tmp.at<float>(0,0) = k1;
	cvD_tmp.at<float>(0,1) = k2;
	cvD_tmp.at<float>(0,2) = p1;
	cvD_tmp.at<float>(0,3) = p2;
	cvD_tmp.at<float>(0,4) = k3;

	if(stereo_cam_->getLeftCamera() == nullptr) 
        throw std::runtime_error("cam_left_ is not allocated.");

	stereo_cam_->getLeftCamera()->initParams(cols, rows, cvK_tmp, cvD_tmp);

	std::cout <<"LEFT  CAMERA PARAMETERS:\n";
	std::cout << "fx_l: " << stereo_cam_->getLeftCamera()->fx() <<", "
			  << "fy_l: " << stereo_cam_->getLeftCamera()->fy() <<", "
			  << "cx_l: " << stereo_cam_->getLeftCamera()->cx() <<", "
			  << "cy_l: " << stereo_cam_->getLeftCamera()->cy() <<", "
			  << "cols_l: " << stereo_cam_->getLeftCamera()->cols() <<", "
			  << "rows_l: " << stereo_cam_->getLeftCamera()->rows() <<"\n";
// Right camera
	rows = fs["Camera.right.height"];	cols = fs["Camera.right.width"];

	fx = fs["Camera.right.fx"];	fy = fs["Camera.right.fy"];
	cx = fs["Camera.right.cx"];	cy = fs["Camera.right.cy"];

	k1 = fs["Camera.right.k1"];	k2 = fs["Camera.right.k2"];	k3 = fs["Camera.right.k3"];
	p1 = fs["Camera.right.p1"];	p2 = fs["Camera.right.p2"];

	cvK_tmp = cv::Mat(3,3,CV_32FC1);
	cvK_tmp.at<float>(0,0) = fx;	cvK_tmp.at<float>(0,1) = 0.0f;	cvK_tmp.at<float>(0,2) = cx;
	cvK_tmp.at<float>(1,0) = 0.0f;	cvK_tmp.at<float>(1,1) = fy;	cvK_tmp.at<float>(1,2) = cy;
	cvK_tmp.at<float>(2,0) = 0.0f;	cvK_tmp.at<float>(2,1) = 0.0f;	cvK_tmp.at<float>(2,2) = 1.0f;
	
	cvD_tmp = cv::Mat(1,5,CV_32FC1);
	cvD_tmp.at<float>(0,0) = k1;
	cvD_tmp.at<float>(0,1) = k2;
	cvD_tmp.at<float>(0,2) = p1;
	cvD_tmp.at<float>(0,3) = p2;
	cvD_tmp.at<float>(0,4) = k3;

	if(stereo_cam_->getRightCamera() == nullptr) 
        throw std::runtime_error("cam_right_ is not allocated.");

	stereo_cam_->getRightCamera()->initParams(cols, rows, cvK_tmp, cvD_tmp);

	std::cout <<"RIGHT CAMERA PARAMETERS:\n";
	std::cout << "fx_r: " << stereo_cam_->getRightCamera()->fx() <<", "
			  << "fy_r: " << stereo_cam_->getRightCamera()->fy() <<", "
			  << "cx_r: " << stereo_cam_->getRightCamera()->cx() <<", "
			  << "cy_r: " << stereo_cam_->getRightCamera()->cy() <<", "
			  << "cols_r: " << stereo_cam_->getRightCamera()->cols() <<", "
			  << "rows_r: " << stereo_cam_->getRightCamera()->rows() <<"\n";


	cv::Mat cvT_lr_tmp = cv::Mat(4,4,CV_32FC1);
	PoseSE3 T_lr;
    fs["T_lr"] >> cvT_lr_tmp;
	for(int i = 0; i < 4; ++i)
		for(int j = 0; j < 4; ++j)
			T_lr(i,j) = cvT_lr_tmp.at<float>(i,j);

	stereo_cam_->setStereoPoseLeft2Right(T_lr);

	std::cout <<"Stereo pose (left to right) T_lr:\n" << T_lr << std::endl;

	stereo_cam_->initStereoCameraToRectify();
	
	std::cout << " - 'loadStereoCameraIntrinsics()' - loaded.\n";
};
