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

	FramePtr frame_left_curr    = std::make_shared<Frame>(cam_rect, img_left_undist, timestamp);
	FramePtr frame_right_curr   = std::make_shared<Frame>(cam_rect, img_right_undist, timestamp, true, frame_left_curr);
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
        const PoseSE3& T_lr = stereo_cam_->getRectifiedStereoPoseLeft2Right();
        const PoseSE3& T_rl = stereo_cam_->getRectifiedStereoPoseRight2Left();

        const cv::Mat& I0_left  = stframe_prev_->getLeftFrame()->getImage();
        const cv::Mat& I0_right = stframe_prev_->getRightFrame()->getImage();
        
        const cv::Mat& I1_left  = stframe_curr->getLeftFrame()->getImage();
        const cv::Mat& I1_right = stframe_curr->getRightFrame()->getImage();

        // [7-2] 직전 tracking 결과를 가져옴
        StereoLandmarkTracking lmtrack_prev;
        lmtrack_prev.pts_left_0  = stframe_prev_->getLeftFrame()->getPtsSeen();
        lmtrack_prev.pts_right_0 = stframe_prev_->getRightFrame()->getPtsSeen();
        lmtrack_prev.pts_left_1  = PixelVec(); // tracking result
        lmtrack_prev.pts_right_1 = PixelVec();
        lmtrack_prev.lms = stframe_prev_->getLeftFrame()->getRelatedLandmarkPtr();
        lmtrack_prev.n_pts = lmtrack_prev.pts_left_0.size();

        int n_pts_exist = lmtrack_prev.n_pts;
        std::cout << " --- # previous pixels : " << n_pts_exist << "\n";

        // [7-3] 현재 stereo frame의 자세를 업데이트한다.
        const PoseSE3& T_wc_prior = stframe_prev_->getLeftFrame()->getPose();
        PoseSE3 T_cw_prior = geometry::inverseSE3_f(T_wc_prior);
        stframe_curr->getLeftFrame()->setPose(T_wc_prior);
        stframe_curr->getRightFrame()->setPose(T_wc_prior * T_lr); // lower cam 에 대한 자세 .. 
        
        // [7-4] tracking 선험정보 계산: pts_l1_prior, pts_u1_prior.
        MaskVec  mask_prior(n_pts_exist,true);
        lmtrack_prev.pts_left_1.resize(n_pts_exist);
        lmtrack_prev.pts_right_1.resize(n_pts_exist);

        for(int i = 0; i < n_pts_exist; ++i)
        {
            const LandmarkPtr& lm = lmtrack_prev.lms[i]; 
            if(lm->isTriangulated() )
            {
                const Point& Xw_prior = lm->get3DPoint();
                Point X_left_1_prior  = T_cw_prior.block<3,3>(0,0)*Xw_prior + T_cw_prior.block<3,1>(0,3);
                Point X_right_1_prior = T_rl.block<3,3>(0,0)*X_left_1_prior + T_rl.block<3,1>(0,3);
                
                Pixel& pt_l1 = lmtrack_prev.pts_left_1[i];
                Pixel& pt_r1 = lmtrack_prev.pts_right_1[i];
        
                // Project prior point.
                pt_l1 = cam_rect->projectToPixel(X_left_1_prior);
                pt_r1 = cam_rect->projectToPixel(X_right_1_prior);
                if( !cam_rect->inImage(pt_l1) || !cam_rect->inImage(pt_r1) )
                { 
                    // out of image.
                    pt_l1 = lmtrack_prev.pts_left_0[i];
                    pt_r1 = lmtrack_prev.pts_right_0[i];
                    mask_prior[i] = false;
                }
            }
        }

        // Track pts_left_0 --> pts_left_1 (lmtrack_l0l1)

        // Track pts_left_1 --> pts_right_1 (lmtrack_l1r1)

        // Check pts_right_1

        // [7-8] Update observations for surviving landmarks
        // for(int i = 0; i < lmtrack_l1r1.pts_left_1.size(); ++i)
        // {
        //     const LandmarkPtr& lm = lmtrack_l1r1.lms[i];
        //     lmtrack_l1r1.lms[i]->addObservationAndRelatedFrame(lmtrack_l1r1.pts_left_1[i], stframe_curr->getLeftFrame());
        //     lmtrack_l1r1.lms[i]->addObservationAndRelatedFrame(lmtrack_l1r1.pts_right_1[i], stframe_curr->getRightFrame());
        // }  

        // if(1) 
        // {
        //     cv::Mat img_color;
        //     I_u1.convertTo(img_color, CV_8UC1);
        //     cv::cvtColor(img_color, img_color, CV_GRAY2RGB);
        //     for(auto pt : lmtrack_l1u1.pts_u1)
        //         cv::circle(img_color, pt, 4, cv::Scalar(255,0,255));
            
        //     for(int i = 0; i < pts_u1_static.size(); ++i)
        //     {
        //         const Pixel& pt = pts_u1_static[i];
        //         if( mask_track_l1u1[i] ) 
        //             cv::circle(img_color, pt, 6, cv::Scalar(0,255,0));
        //     }
        //     std::stringstream ss;
        //     ss << n << " - feature_backtrack";
        //     cv::namedWindow(ss.str());
        //     cv::imshow(ss.str(), img_color);
        //     cv::waitKey(0);
        // }


        // TODO
        // [7-9] Motion Estimation via pose-only BA (stereo version)
        // Using landmarks with 3D point, stereo pose-only BA.
        // pts_left_1 , pts_right_1 , Xw , T_cw_prior    : needed.


        // [7-9] Extract new points from empty bins.
        PixelVec pts_left_1_new;
        // this->extractor_->updateWeightBin(lmtrack_l1u1.pts_l1);
        // this->extractor_->extractORBwithBinning_fast(I1_left, pts_left_1_new, true);
        int n_pts_new = pts_left_1_new.size();

        if(n_pts_new > 0)
        {
            // If there are new points, add it.
            std::cout << " --- --- # of NEW points : " << n_pts_new  << "\n";

            // Track static stereo
            MaskVec mask_new;
            PixelVec pts_right_1_new;
            this->tracker_->trackBidirection(
                I1_left, I1_right, pts_left_1_new, 
                params_.feature_tracker.window_size, params_.feature_tracker.max_level, params_.feature_tracker.thres_error, params_.feature_tracker.thres_bidirection,
                pts_right_1_new, mask_new); // lmtrack_curr.pts_u1 에 prior pixels가 이미 들어있다.

            // 새로운 특징점은 새로운 landmark가 된다.
            for(int i = 0; i < pts_right_1_new.size(); ++i)
            {
                if( mask_new[i] )
                {
                    const Pixel& pt_l_new = pts_left_1_new[i];
                    const Pixel& pt_r_new = pts_right_1_new[i];

                    LandmarkPtr lmptr = std::make_shared<Landmark>(pt_l_new, stframe_curr->getLeftFrame(), cam_rect);
                    lmtrack_l1r1.pts_left_1.push_back(pt_l_new);
                    lmtrack_l1r1.pts_right_1.push_back(pt_r_new);
                    lmtrack_l1r1.lms.push_back(lmptr);
                    lmptr->set3DPoint(Xw_prior);

                    lmptr->addObservationAndRelatedFrame(pt_r_new, stframe_curr->getRightFrame());
                }
            }
        }
        else
            std::cout << " --- --- NO NEW POINTS.\n";

        // [7-10] Update tracking information (set related pixels and landmarks)
        stframe_curr->getLeftFrame()->setPtsSeenAndRelatedLandmarks(lmtrack_l1r1.pts_left_1, lmtrack_l1r1.lms);
        stframe_curr->getRightFrame()->setPtsSeenAndRelatedLandmarks(lmtrack_l1r1.pts_right_1, lmtrack_l1r1.lms);


        // Keyframe?


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
        const PoseSE3& T_rl = stereo_cam_->getRectifiedStereoPoseRight2Left();
        const Rot3& R_rl = T_rl.block<3,3>(0,0);
        const Pos3& t_rl = T_rl.block<3,1>(0,3);
        for(int i = 0; i < lmtrack_staticklt.n_pts; ++i)
        {
            const Pixel& pt0 = lmtrack_staticklt.pts_left_1[i];
            const Pixel& pt1 = lmtrack_staticklt.pts_right_1[i];
            const LandmarkPtr& lm = lmtrack_staticklt.lms[i];
            
            // Reconstruct points
            Point Xl, Xr;
            Mapping::triangulateDLT(pt0, pt1, R_rl, t_rl, cam_rect, Xl, Xr);

            // Check reprojection error for the first image
            Pixel pt0_proj = cam_rect->projectToPixel(Xl);
            Pixel dpt0 = pt0 - pt0_proj;
            float dpt0_norm2 = dpt0.x*dpt0.x + dpt0.y*dpt0.y;
            if(dpt0_norm2 > 1.0) continue;

            Pixel pt1_proj = cam_rect->projectToPixel(Xr);
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
