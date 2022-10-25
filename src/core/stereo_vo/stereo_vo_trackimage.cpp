#include "core/stereo_vo/stereo_vo.h"

void StereoVO::trackStereoImages(
    const cv::Mat& img_left, const cv::Mat& img_right, const double& timestamp)
{
	float THRES_SAMPSON  = params_.feature_tracker.thres_sampson;
	float THRES_PARALLAX = params_.map_update.thres_parallax;

	// Generate statistics
	AlgorithmStatistics::LandmarkStatistics  statcurr_landmark;
	AlgorithmStatistics::FrameStatistics     statcurr_frame;
	AlgorithmStatistics::KeyframeStatistics  statcurr_keyframe;
	AlgorithmStatistics::ExecutionStatistics statcurr_execution;
			
	// 이미지 undistort (KITTI라서 할 필요 X)
	cv::Mat img_left_undist, img_right_undist;
	if(system_flags_.flagDoUndistortion)
	{
        timer::tic();
        stereo_cam_->rectifyStereoImages(
            img_left, img_right, 
            img_left_undist, img_right_undist);
		// stereo_cam_->getLeftCamera()->undistortImage(img_left, img_left_undist);
		// stereo_cam_->getRightCamera()->undistortImage(img_right, img_right_undist);

		img_left_undist.convertTo(img_left_undist, CV_8UC1);
		img_right_undist.convertTo(img_right_undist, CV_8UC1);
		std::cout << colorcode::text_green << "Time [stereo undistort ]: " << timer::toc(0) << " [ms]\n" << colorcode::cout_reset;
	}
	else 
	{
		img_left.copyTo(img_left_undist);
		img_right.copyTo(img_right_undist);
	}

    // Algorithm implementation
    CameraConstPtr cam_rect = stereo_cam_->getRectifiedCamera(); // In stereo setting, use rectified images.

	FramePtr frame_left_curr = std::make_shared<Frame>(cam_rect, img_left_undist, timestamp);
	FramePtr frame_right_curr = std::make_shared<Frame>(cam_rect, img_right_undist, timestamp, true, frame_left_curr);
	StereoFramePtr stframe_curr = std::make_shared<StereoFrame>(frame_left_curr, frame_right_curr);
	
	this->saveStereoFrame(stframe_curr);   

    if(1)
    {
         std::string window_name = "img_rect_left";
        cv::namedWindow(window_name);
        cv::imshow(window_name, stframe_curr->getLeftFrame()->getImage());

        window_name = "img_rect_right";
        cv::namedWindow(window_name);
        cv::imshow(window_name, stframe_curr->getRightFrame()->getImage());
        cv::waitKey(3);
    }
   

    // Algorithm
    if( this->system_flags_.flagFirstImageGot )
    {
        const cv::Mat& I0_left  = stframe_prev_->getLeftFrame()->getImage();
        const cv::Mat& I0_right = stframe_prev_->getRightFrame()->getImage();
        
        const cv::Mat& I1_left  = stframe_curr->getLeftFrame()->getImage();
        const cv::Mat& I1_right = stframe_curr->getRightFrame()->getImage();



    }
    else 
    {
        // The very first image.
        const cv::Mat& I1_left  = stframe_curr->getLeftFrame()->getImage();
        const cv::Mat& I1_right = stframe_curr->getRightFrame()->getImage();

        // 첫번째 stereo frame의 자세는 아래와 같다. 
        stframe_curr->getLeftFrame()->setPose(PoseSE3::Identity());
        stframe_curr->getRightFrame()->setPose(PoseSE3::Identity()*stereo_cam_->getRectifiedStereoPoseLeft2Right()); // lower cam 에 대한 자세 .. 슬 필요 없다.

        // Extract initial feature points.
        StereoLandmarkTracking lmtrack_curr;
        extractor_->resetWeightBin();
        extractor_->extractORBwithBinning_fast(I1_left, lmtrack_curr.pts_left_1, true);
        std::cout << "# extracted features : " << lmtrack_curr.pts_left_1.size()  << std::endl;

        // 초기 landmark 생성
        lmtrack_curr.lms.resize(lmtrack_curr.pts_left_1.size());
        for(int i = 0; i < lmtrack_curr.pts_left_1.size(); ++i)
        {
            const Pixel& pt_left = lmtrack_curr.pts_left_1[i];
            LandmarkPtr lmptr = std::make_shared<Landmark>(pt_left, stframe_curr->getLeftFrame(), cam_rect);
            lmtrack_curr.lms[i] = lmptr;
        } // Now, 'pts_left_1', 'lms' are filled. 'pts_right_1' should be filled.

        // make dummy pixels
        lmtrack_curr.pts_left_0.resize(lmtrack_curr.pts_left_1.size());
        lmtrack_curr.pts_right_0.resize(lmtrack_curr.pts_left_1.size());
        lmtrack_curr.lms.resize(lmtrack_curr.pts_left_1.size());

        // Feature tracking.
        // 1) Track 'static stereo' (I1_left --> I1_right)
        timer::tic();
        MaskVec mask_track;
        this->tracker_->trackBidirection(I1_left, I1_right,
            lmtrack_curr.pts_left_1, params_.feature_tracker.window_size, params_.feature_tracker.max_level, params_.feature_tracker.thres_error, params_.feature_tracker.thres_bidirection,
            lmtrack_curr.pts_right_1, mask_track); // lmtrack_curr.pts_u1 에 prior pixels가 이미 들어있다.

        StereoLandmarkTracking lmtrack_staticklt(lmtrack_curr, mask_track);
		std::cout << colorcode::text_green << "Time [track bidirection]: " << timer::toc(0) << " [ms]\n" << colorcode::cout_reset;       

        // make landmarks
        for(int i = 0; i < lmtrack_staticklt.n_pts; ++i)
        {
            const Pixel& pt_left  = lmtrack_staticklt.pts_left_1[i];
            const Pixel& pt_right = lmtrack_staticklt.pts_right_1[i];
            LandmarkPtr lmptr = std::make_shared<Landmark>(pt_left, stframe_curr->getLeftFrame(), stereo_cam_->getLeftCamera());
            lmptr->addObservationAndRelatedFrame(pt_right, stframe_curr->getRightFrame());

            lmtrack_staticklt.lms[i] = lmptr;

            this->saveLandmark(lmptr);                        
        }
        std::cout << "# static klt success pts : " << lmtrack_staticklt.n_pts  << std::endl;

        // 3D reconstruction 
        int cnt_recon = 0;
        const PoseSE3& T_rl = stereo_cam_->getStereoPoseRight2Left();
        const Rot3& R_rl = T_rl.block<3,3>(0,0);
        const Pos3& t_rl = T_rl.block<3,1>(0,3);
        for(int i = 0; i < lmtrack_staticklt.n_pts; ++i)
        {
            const Pixel& pt0 = lmtrack_staticklt.pts_left_1[i];
            const Pixel& pt1 = lmtrack_staticklt.pts_right_1[i];
            const LandmarkPtr& lm = lmtrack_staticklt.lms[i];
            // Reconstruct points
            Point Xl, Xr;
            Mapping::triangulateDLT(pt0, pt1, R_rl, t_rl, stereo_cam_->getLeftCamera(),stereo_cam_->getRightCamera(), Xl, Xr);

            // Check reprojection error for the first image
            Pixel pt0_proj = stereo_cam_->getLeftCamera()->projectToPixel(Xl);
            Pixel dpt0 = pt0 - pt0_proj;
            float dpt0_norm2 = dpt0.x*dpt0.x + dpt0.y*dpt0.y;
            if(dpt0_norm2 > 1.0) continue;

            Pixel pt1_proj = stereo_cam_->getRightCamera()->projectToPixel(Xr);
            Pixel dpt1 = pt1 - pt1_proj;
            float dpt1_norm2 = dpt1.x*dpt1.x + dpt1.y*dpt1.y;
            if(dpt1_norm2 > 1.0) continue;

            // Check the point in front of cameras
            if(Xl(2) > 0 && Xr(2) > 0) 
            {
                Point Xworld = Xl;
                lm->set3DPoint(Xworld);
                ++cnt_recon;
            }
        }
        std::cout << "# of reconstructed points: " << cnt_recon << " / " << lmtrack_staticklt.n_pts << std::endl;
        
        // Set related pixels and landmarks    
        stframe_curr->getLeftFrame()->setPtsSeenAndRelatedLandmarks(lmtrack_staticklt.pts_left_1, lmtrack_staticklt.lms);
        stframe_curr->getRightFrame()->setPtsSeenAndRelatedLandmarks(lmtrack_staticklt.pts_right_1, lmtrack_staticklt.lms);


         this->system_flags_.flagFirstImageGot = true;

        ROS_INFO_STREAM("============ Starts to iterate all images... ============");
        
    }

// [6] Update prev
	this->stframe_prev_ = stframe_curr;	
};
