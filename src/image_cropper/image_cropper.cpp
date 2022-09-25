#include "image_cropper/image_cropper.h"

ImageCropper::ImageCropper(ros::NodeHandle& nh)
: nh_(nh), it_(nh_)
{
    // get parameters from launch file
    this->getParameters();

    if(n_cameras_ > 0){
        for(int i = 0; i < n_cameras_; ++i) {
            std::string name_temp = topicnames_image_raw_[i];
            topicnames_image_crop_.push_back(topicnames_image_raw_[i] + "_cropped");
            this->subs_imgs_.push_back(this->it_.subscribe(topicnames_image_raw_[i], 1, boost::bind(&ImageCropper::callbackImage, this, _1, i)));
            this->buf_imgs_.push_back(cv::Mat());
            this->buf_imgs_cropped_.push_back(cv::Mat());
        }

        for(int i = 0; i < n_cameras_; ++i){
            pubs_cropped_.emplace_back(nh_.advertise<sensor_msgs::Image>(topicnames_image_crop_[i],1));
        }
    }
    else
    {
        throw std::runtime_error("n_cameras <= 0!");
    }

    // run
    this->run();
};

ImageCropper::~ImageCropper()
{
    
};

void ImageCropper::getParameters()
{
    if(!ros::param::has("~image_name_0"))
        throw std::runtime_error("'image_name_0' is not set.");
    if(!ros::param::has("~image_name_1"))
        throw std::runtime_error("'image_name_1' is not set.");
    if(!ros::param::has("~image_name_2"))
        throw std::runtime_error("'image_name_2' is not set.");
    if(!ros::param::has("~n_cameras"))
        throw std::runtime_error("'n_cameras' is not set.");

    ros::param::get("~n_cameras", n_cameras_);

    topicnames_image_raw_.resize(n_cameras_);
    
    ros::param::get("~image_name_0", topicnames_image_raw_[0]);
    ros::param::get("~image_name_1", topicnames_image_raw_[1]);
    ros::param::get("~image_name_2", topicnames_image_raw_[2]);

    if(!ros::param::has("~img0_left_crop"))
        throw std::runtime_error("'img0_left_crop' is not set.");
    if(!ros::param::has("~img0_right_crop"))
        throw std::runtime_error("'img0_right_crop' is not set.");
    if(!ros::param::has("~img0_top_crop"))
        throw std::runtime_error("'img0_top_crop' is not set.");
    if(!ros::param::has("~img0_bottom_crop"))
        throw std::runtime_error("'img0_bottom_crop' is not set.");

    if(!ros::param::has("~img1_left_crop"))
        throw std::runtime_error("'img1_left_crop' is not set.");
    if(!ros::param::has("~img1_right_crop"))
        throw std::runtime_error("'img1_right_crop' is not set.");
    if(!ros::param::has("~img1_top_crop"))
        throw std::runtime_error("'img1_top_crop' is not set.");
    if(!ros::param::has("~img1_bottom_crop"))
        throw std::runtime_error("'img1_bottom_crop' is not set.");

    if(!ros::param::has("~img2_left_crop"))
        throw std::runtime_error("'img2_left_crop' is not set.");
    if(!ros::param::has("~img2_right_crop"))
        throw std::runtime_error("'img2_right_crop' is not set.");
    if(!ros::param::has("~img2_top_crop"))
        throw std::runtime_error("'img2_top_crop' is not set.");
    if(!ros::param::has("~img2_bottom_crop"))
        throw std::runtime_error("'img2_bottom_crop' is not set.");

    crop_size_.resize(n_cameras_);

    crop_size_[0].n_cols = 1032;
    crop_size_[0].n_rows = 772;
    ros::param::get("~img0_left_crop", crop_size_[0].crop_left);
    ros::param::get("~img0_right_crop", crop_size_[0].crop_right);
    ros::param::get("~img0_top_crop", crop_size_[0].crop_top);
    ros::param::get("~img0_bottom_crop", crop_size_[0].crop_bottom);
    crop_size_[0].n_cols_crop = crop_size_[0].n_cols-(crop_size_[0].crop_left+crop_size_[0].crop_right);
    crop_size_[0].n_rows_crop = crop_size_[0].n_rows-(crop_size_[0].crop_top+crop_size_[0].crop_bottom);

    crop_size_[0].roi = cv::Rect(cv::Point(crop_size_[0].crop_left, crop_size_[0].crop_top),
                        cv::Point(crop_size_[0].n_cols-crop_size_[0].crop_right, crop_size_[0].n_rows - crop_size_[0].crop_bottom));
    

    crop_size_[1].n_cols = 1032;
    crop_size_[1].n_rows = 772;
    ros::param::get("~img1_left_crop", crop_size_[1].crop_left);
    ros::param::get("~img1_right_crop", crop_size_[1].crop_right);
    ros::param::get("~img1_top_crop", crop_size_[1].crop_top);
    ros::param::get("~img1_bottom_crop", crop_size_[1].crop_bottom);
    crop_size_[1].n_cols_crop = crop_size_[1].n_cols-(crop_size_[1].crop_left+crop_size_[1].crop_right);
    crop_size_[1].n_rows_crop = crop_size_[1].n_rows-(crop_size_[1].crop_top+crop_size_[1].crop_bottom);
    crop_size_[1].roi = cv::Rect(cv::Point(crop_size_[1].crop_left, crop_size_[1].crop_top),
                        cv::Point(crop_size_[1].n_cols-crop_size_[1].crop_right, crop_size_[1].n_rows - crop_size_[1].crop_bottom));
    
    crop_size_[2].n_cols = 1032;
    crop_size_[2].n_rows = 772;
    ros::param::get("~img2_left_crop", crop_size_[2].crop_left);
    ros::param::get("~img2_right_crop", crop_size_[2].crop_right);
    ros::param::get("~img2_top_crop", crop_size_[2].crop_top);
    ros::param::get("~img2_bottom_crop", crop_size_[2].crop_bottom);
    crop_size_[2].n_cols_crop = crop_size_[2].n_cols-(crop_size_[2].crop_left+crop_size_[2].crop_right);
    crop_size_[2].n_rows_crop = crop_size_[2].n_rows-(crop_size_[2].crop_top+crop_size_[2].crop_bottom);
    crop_size_[2].roi = cv::Rect(cv::Point(crop_size_[2].crop_left, crop_size_[2].crop_top),
                        cv::Point(crop_size_[2].n_cols-crop_size_[2].crop_right, crop_size_[2].n_rows - crop_size_[2].crop_bottom));
    

};

void ImageCropper::callbackImage(const sensor_msgs::ImageConstPtr& msg, const int& id){
    cv_bridge::CvImagePtr cv_ptr;
    cv_ptr = cv_bridge::toCvCopy(msg, msg->encoding); 
    ROS_INFO_STREAM("ImageCropper - " << id << "-th image, encoding : " << msg->encoding);
    cv_ptr->image.copyTo(buf_imgs_[id]);

    // Crop the image
    buf_imgs_[id](crop_size_[id].roi).copyTo(buf_imgs_cropped_[id]);

    std::cout << "size: " << buf_imgs_cropped_[id].size() << std::endl;

    // Publish cropped image
    cv_bridge::CvImage msg_crop;
    msg_crop.header.stamp    = ros::Time::now(); // Same times
    msg_crop.encoding        = "mono8"; // Or whatever
    msg_crop.image           = buf_imgs_cropped_[id]; // Your cv::Mat
    pubs_cropped_[id].publish(msg_crop);
};

void ImageCropper::run()
{
    ROS_INFO_STREAM("ImageCropper - run()");
    ros::Rate rate(5000);
    while(ros::ok()){
        ros::spinOnce();
        rate.sleep();
    }
};
