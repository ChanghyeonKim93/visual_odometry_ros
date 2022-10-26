#include "core/stereo_vo/stereo_vo.h"

void StereoVO::trackStereoImages(
    const cv::Mat& img_left, const cv::Mat& img_right, const double& timestamp)
{
	float THRES_SAMPSON  = params_.feature_tracker.thres_sampson;
	float THRES_PARALLAX = params_.map_update.thres_parallax;

    // Rectified Camera.
    CameraConstPtr& cam_rect = stereo_cam_->getRectifiedCamera(); // In stereo setting, use rectified images.

	// Generate statistics
	AlgorithmStatistics::LandmarkStatistics  statcurr_landmark;
	AlgorithmStatistics::FrameStatistics     statcurr_frame;
	AlgorithmStatistics::KeyframeStatistics  statcurr_keyframe;
	AlgorithmStatistics::ExecutionStatistics statcurr_execution;
			
	// 이미지 undistort (KITTI라서 할 필요 X)
	cv::Mat img_left_undist, img_right_undist;
	if( system_flags_.flagDoUndistortion )
	{
        timer::tic();
        stereo_cam_->rectifyStereoImages(img_left, img_right, 
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
    // Make new stereo frame for current images.
	StereoFramePtr stframe_curr = std::make_shared<StereoFrame>(img_left_undist, img_right_undist, cam_rect, cam_rect, timestamp);
	
	this->saveStereoFrame(stframe_curr);      

    // Algorithm
    if( this->system_flags_.flagFirstImageGot )
    {
        // 초기화가 된 상태에서 반복적 구동.
        // Stereo static pose 가져오기.
        const PoseSE3& T_lr = stereo_cam_->getRectifiedStereoPoseLeft2Right();
        const PoseSE3& T_rl = stereo_cam_->getRectifiedStereoPoseRight2Left();

        // 이전 이미지 가져오기
        const cv::Mat& I0_left  = stframe_prev_->getLeft()->getImage();
        const cv::Mat& I0_right = stframe_prev_->getRight()->getImage();
        
        // 현재 이미지 가져오기
        const cv::Mat& I1_left  = stframe_curr->getLeft()->getImage();
        const cv::Mat& I1_right = stframe_curr->getRight()->getImage();

        // [1] 직전 tracking 결과를 가져옴
        StereoLandmarkTracking lmtrack_prev;
        lmtrack_prev.pts_l0 = stframe_prev_->getLeft()->getPtsSeen();
        lmtrack_prev.pts_r0 = stframe_prev_->getRight()->getPtsSeen();
        lmtrack_prev.n_pts  = lmtrack_prev.pts_l0.size();
        lmtrack_prev.pts_l1 = PixelVec(lmtrack_prev.n_pts); // tracking result
        lmtrack_prev.pts_r1 = PixelVec(lmtrack_prev.n_pts);
        lmtrack_prev.lms    = stframe_prev_->getLeft()->getRelatedLandmarkPtr();

        // 사이즈 비교
        std::cout << " size: lmtrack_prev : " 
            << lmtrack_prev.pts_l0.size() << ", " 
            << lmtrack_prev.pts_r0.size() << ", " 
            << lmtrack_prev.pts_l1.size() << ", " 
            << lmtrack_prev.pts_r1.size() << ", " 
            << lmtrack_prev.lms.size() << "\n"; 

        int n_pts_exist = lmtrack_prev.n_pts;
        std::cout << " --- # previous pixels : " << n_pts_exist << "\n";
        
        // [2] 현재 stereo frame의 자세를 업데이트한다.
        // 이전 자세와 이전 업데이트 값을 가져온다. (constant velocity assumption)
        const PoseSE3& T_wp = stframe_prev_->getLeft()->getPose();
        const PoseSE3& T_pw = stframe_prev_->getLeft()->getPoseInv();
        const PoseSE3& dT_pc_prev = stframe_prev_->getLeft()->getPoseDiff01();

        PoseSE3 T_wc_prior = T_wp * dT_pc_prev;
        PoseSE3 T_cw_prior = geometry::inverseSE3_f(T_wc_prior);

        std::cout << "T_wc_prior:\n" << T_wc_prior << std::endl;

        // [3] tracking을 위해, 'pts_l1' 'pts_r1' 에 대한 prior 계산.
        MaskVec  mask_prior(n_pts_exist, true);
        for(int i = 0; i < n_pts_exist; ++i)
        {
            const LandmarkPtr& lm = lmtrack_prev.lms[i]; 
            if( lm->isTriangulated() )
            {
                const Point& X = lm->get3DPoint();
                Point X_l1 = T_cw_prior.block<3,3>(0,0)*X + T_cw_prior.block<3,1>(0,3);
                Point X_r1 = T_rl.block<3,3>(0,0)*X_l1 + T_rl.block<3,1>(0,3);
                
                Pixel& pt_l1 = lmtrack_prev.pts_l1[i];
                Pixel& pt_r1 = lmtrack_prev.pts_r1[i];
        
                // Project prior point.
                pt_l1 = cam_rect->projectToPixel(X_l1);
                pt_r1 = cam_rect->projectToPixel(X_r1);
                if( !cam_rect->inImage(pt_l1) || !cam_rect->inImage(pt_r1) 
                    || X_l1(2) < 0.1 || X_r1(2) < 0.1 )
                { 
                    // out of image.
                    pt_l1 = lmtrack_prev.pts_l0[i];
                    pt_r1 = lmtrack_prev.pts_r0[i];
                    mask_prior[i] = false;
                }
            }
            else   
            {
                lmtrack_prev.pts_l1[i] = lmtrack_prev.pts_l0[i];
                lmtrack_prev.pts_r1[i] = lmtrack_prev.pts_r0[i];
            }
        }

        /* Tracking order
            l0 -> l1
            l1 -> r1
            // r0 -> r1
            // check : r1 == r1
        */
        // [4] Track l0 --> l1 (lmtrack_l0l1)
        timer::tic();
        MaskVec mask_l0l1;
        this->tracker_->trackWithPrior(I0_left, I1_left,
            lmtrack_prev.pts_l0, params_.feature_tracker.window_size, params_.feature_tracker.max_level, params_.feature_tracker.thres_error,
            lmtrack_prev.pts_l1, mask_l0l1); // lmtrack_curr.pts_u1 에 prior pixels가 이미 들어있다.

        StereoLandmarkTracking lmtrack_l0l1(lmtrack_prev, mask_l0l1);
        std::cout << "# pts : " << lmtrack_l0l1.n_pts << std::endl;
		std::cout << colorcode::text_green << "Time [track l0l1]: " << timer::toc(0) << " [ms]\n" << colorcode::cout_reset;       

        // [5] Track l1 --> r1 (lmtrack_l1r1)
        timer::tic();
        MaskVec mask_l1r1;
        this->tracker_->trackWithPrior(I1_left, I1_right,
            lmtrack_l0l1.pts_l0, params_.feature_tracker.window_size, params_.feature_tracker.max_level, params_.feature_tracker.thres_error,
            lmtrack_l0l1.pts_r1, mask_l1r1); // lmtrack_curr.pts_u1 에 prior pixels가 이미 들어있다.

        StereoLandmarkTracking lmtrack_kltok(lmtrack_l0l1, mask_l1r1);
        std::cout << "# pts : " << lmtrack_kltok.n_pts << std::endl;
		std::cout << colorcode::text_green << "Time [track l1r1]: " << timer::toc(0) << " [ms]\n" << colorcode::cout_reset;

        // [6] Check sampson distance 0.01 ms
        timer::tic();
        std::vector<float> symm_epi_dist;
        motion_estimator_->calcSampsonDistance(lmtrack_kltok.pts_l1, lmtrack_kltok.pts_r1, cam_rect, T_rl.block<3,3>(0,0), T_rl.block<3,1>(0,3), symm_epi_dist);
        
        MaskVec mask_sampson(lmtrack_kltok.pts_l1.size(), true);
        for(int i = 0; i < mask_sampson.size(); ++i)
            mask_sampson[i] = (symm_epi_dist[i] < THRES_SAMPSON*5);
        
        StereoLandmarkTracking lmtrack_final(lmtrack_kltok, mask_sampson);
        std::cout << "# pts : " << lmtrack_final.n_pts << std::endl;
		std::cout << colorcode::text_green << "Time [sampson ]: " << timer::toc(0) << " [ms]\n" << colorcode::cout_reset;


        // [7] Motion Estimation via pose-only BA (stereo version)
        // Using landmarks with 3D point, stereo pose-only BA.
        // pts_left_1 , pts_right_1 , Xw , T_cw_prior    : needed. 
        //    --> return: T_wc
        timer::tic();
        MaskVec mask_motion(lmtrack_final.pts_l1.size(), true); 

        // Find point with 3D points
        PoseSE3 dT_pc_poBA = dT_pc_prev;

        PoseSE3 T_wc = T_wc_prior;

        int cnt_poBA = 0;
        PointVec Xp_poBA;
        PixelVec pts_l1_poBA;
        PixelVec pts_r1_poBA;
        MaskVec mask_poBA;
        std::vector<int> index_poBA;
        for(int i = 0; i < lmtrack_final.pts_l1.size(); ++i)
        {
            LandmarkConstPtr& lm = lmtrack_final.lms[i];

            if( lm->isTriangulated() )
            {   
                const Pixel& pt_l1 = lmtrack_final.pts_l1[i];
                const Pixel& pt_r1 = lmtrack_final.pts_r1[i];
                const Point& X     = lm->get3DPoint();

                Point Xp = T_pw.block<3,3>(0,0) * X + T_pw.block<3,1>(0,3);

                Xp_poBA.push_back(Xp);
                pts_l1_poBA.push_back(pt_l1);
                pts_r1_poBA.push_back(pt_r1);
                mask_poBA.push_back(true);
                index_poBA.push_back(i);
                ++cnt_poBA;
            }
        }
        std::cout << "  size comparison : " 
            << Xp_poBA.size() << ", "
            << pts_l1_poBA.size() << ", "
            << pts_r1_poBA.size() << ", "
            << mask_poBA.size() << ", "
            << index_poBA.size() << "\n ";
        std::cout << "  # of poseonly BA point: " << cnt_poBA << std::endl;

        bool poseonlyBA_success =
            motion_estimator_->poseOnlyBundleAdjustment_Stereo(
                Xp_poBA, pts_l1_poBA, pts_r1_poBA, cam_rect, cam_rect, T_lr, params_.motion_estimator.thres_poseba_error,
                dT_pc_poBA, mask_poBA);

        if( !poseonlyBA_success )
        {
            throw std::runtime_error("PoseOnlyStereoBA is failed!");
        }
        else{
            // Success. Update pose.
            for(int i = 0; i < mask_poBA.size(); ++i)
            {
                const int& idx = index_poBA[i];
                if( mask_poBA[i] )
                    mask_motion[idx] = true;
                else 
                    mask_motion[idx] = false;
            }

            T_wc = T_wp * dT_pc_poBA;

            stframe_curr->setStereoPoseByLeft(T_wc, T_lr);
            stframe_curr->getLeft()->setPoseDiff10(dT_pc_poBA.inverse());
        }

        StereoLandmarkTracking lmtrack_motion_ok(lmtrack_final, mask_motion);
		std::cout << colorcode::text_green << "Time [motion est]: " << timer::toc(0) << " [ms]\n" << colorcode::cout_reset;

        // [8] Reconstruction
        int cnt_recon = 0;
        const Rot3& R_rl = T_rl.block<3,3>(0,0);
        const Pos3& t_rl = T_rl.block<3,1>(0,3);
        for(int i = 0; i < lmtrack_motion_ok.n_pts; ++i)
        {
            const Pixel&      pt0 = lmtrack_motion_ok.pts_l1[i];
            const Pixel&      pt1 = lmtrack_motion_ok.pts_r1[i];
            const LandmarkPtr& lm = lmtrack_motion_ok.lms[i];
            
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
                Point Xworld = T_wc.block<3,3>(0,0)* Xl  + T_wc.block<3,1>(0,3);
                lm->set3DPoint(Xworld);
                ++cnt_recon;
            }
        }
        std::cout << "# of reconstructed points: " << cnt_recon << " / " << lmtrack_motion_ok.n_pts << std::endl;
        

        // [9] Update observations for surviving landmarks
        for(int i = 0; i < lmtrack_motion_ok.pts_l1.size(); ++i)
        {
            const LandmarkPtr& lm = lmtrack_motion_ok.lms[i];
            lm->addObservationAndRelatedFrame(lmtrack_motion_ok.pts_l1[i], stframe_curr->getLeft());
            lm->addObservationAndRelatedFrame(lmtrack_motion_ok.pts_r1[i], stframe_curr->getRight());
        }  

        if(0) 
        {
            cv::Mat img_color;
            I1_right.convertTo(img_color, CV_8UC1);
            cv::cvtColor(img_color, img_color, CV_GRAY2RGB);
            for(const auto& pt : lmtrack_motion_ok.pts_r1)
                cv::circle(img_color, pt, 5, cv::Scalar(0,255,0));
            
            std::stringstream ss;
            ss << "track r1";
            cv::namedWindow(ss.str());
            cv::imshow(ss.str(), img_color);


            I1_left.convertTo(img_color, CV_8UC1);
            cv::cvtColor(img_color, img_color, CV_GRAY2RGB);
            for(const auto& pt : lmtrack_motion_ok.pts_l1)
                cv::circle(img_color, pt, 5, cv::Scalar(0,255,0));
            
            ss.clear();
            ss.flush();
            ss << "track l1";
            cv::namedWindow(ss.str());
            cv::imshow(ss.str(), img_color);
            cv::waitKey(0);
        }

        // [10] Extract new points from empty bins.
        PixelVec pts_l1_new;
        extractor_->updateWeightBin(lmtrack_motion_ok.pts_l1);
        extractor_->extractORBwithBinning_fast(I1_left, pts_l1_new, true);
        int n_pts_new = pts_l1_new.size();

        std::cout << "# of newly extracted pts on empty space: " << n_pts_new << std::endl; 

        if(n_pts_new > 0)
        {
            // If there are new points, add it.
            std::cout << " --- --- # of NEW points : " << n_pts_new  << "\n";

            // Track static stereo
            MaskVec mask_new;
            PixelVec pts_r1_new;
            this->tracker_->trackBidirection(
                I1_left, I1_right, pts_l1_new, 
                params_.feature_tracker.window_size, params_.feature_tracker.max_level, params_.feature_tracker.thres_error, params_.feature_tracker.thres_bidirection,
                pts_r1_new, mask_new); // lmtrack_curr.pts_u1 에 prior pixels가 이미 들어있다.

            // 새로운 특징점은 새로운 landmark가 된다.
            for(int i = 0; i < pts_r1_new.size(); ++i)
            {
                if( mask_new[i] )
                {
                    const Pixel& pt_l1_new = pts_l1_new[i];
                    const Pixel& pt_r1_new = pts_r1_new[i];

                    // Reconstruct 3D point
                    Point Xl, Xr;
                    Mapping::triangulateDLT(pt_l1_new, pt_r1_new, T_rl.block<3,3>(0,0), T_rl.block<3,1>(0,3), cam_rect, Xl, Xr);

                    if( Xl(2) > 0 && Xr(2) > 0)
                    {
                        Point Xw = T_wc.block<3,3>(0,0) * Xl + T_wc.block<3,1>(0,3);
                    
                        LandmarkPtr lmptr = std::make_shared<Landmark>(pt_l1_new, stframe_curr->getLeft(), cam_rect);
                        lmptr->addObservationAndRelatedFrame(pt_r1_new, stframe_curr->getRight());

                        lmtrack_motion_ok.pts_l1.push_back(pt_l1_new);
                        lmtrack_motion_ok.pts_r1.push_back(pt_r1_new);
                        lmtrack_motion_ok.lms.push_back(lmptr);

                        lmptr->set3DPoint(Xw);
                    }
                    
                }
            }
        }
        else
            std::cout << " --- --- NO NEW POINTS.\n";
        

        // [11] Update tracking information (set related pixels and landmarks)
        std::cout << "# of final landmarks : " << lmtrack_motion_ok.pts_l1.size() << std::endl;
        stframe_curr->setStereoPtsSeenAndRelatedLandmarks(lmtrack_motion_ok.pts_l1, lmtrack_motion_ok.pts_r1, lmtrack_motion_ok.lms);

        // [12] Keyframe selection?
        bool flag_add_new_keyframe = this->stkeyframes_->checkUpdateRule(stframe_curr);

        if( flag_add_new_keyframe )
        {
            // Add keyframe
            this->saveStereoKeyframe(stframe_curr);
            this->stkeyframes_->addNewStereoKeyframe(stframe_curr);
            std::cout << "add done\n";
            

            // Local Bundle Adjustment
            motion_estimator_->localBundleAdjustmentSparseSolver_Stereo(stkeyframes_, stereo_cam_);

timer::tic();
PointVec X_tmp;
const LandmarkPtrVec& lmvec_tmp = stframe_curr->getLeft()->getRelatedLandmarkPtr();
for(int i = 0; i < lmvec_tmp.size(); ++i)
{
	X_tmp.push_back(lmvec_tmp[i]->get3DPoint());
}
statcurr_keyframe.Twc = stframe_curr->getLeft()->getPose();
statcurr_keyframe.mappoints = X_tmp;
stat_.stats_keyframe.push_back(statcurr_keyframe);

for(int j = 0; j < stat_.stats_keyframe.size(); ++j){
	stat_.stats_keyframe[j].Twc = all_stkeyframes_[j]->getLeft()->getPose();

	const LandmarkPtrVec& lmvec_tmp = all_stkeyframes_[j]->getLeft()->getRelatedLandmarkPtr();
	stat_.stats_keyframe[j].mappoints.resize(lmvec_tmp.size());
	for(int i = 0; i < lmvec_tmp.size(); ++i) {
		stat_.stats_keyframe[j].mappoints[i] = lmvec_tmp[i]->get3DPoint();
	}
}
std::cout << colorcode::text_green << "Time [RECORD KEYFR STAT]: " << timer::toc(0) << " [ms]\n" << colorcode::cout_reset;

        } // KEYFRAME addition done.


    }
    else 
    {
        // The very first image.
        const cv::Mat& I1_left  = stframe_curr->getLeft()->getImage();
        const cv::Mat& I1_right = stframe_curr->getRight()->getImage();

        // 첫번째 stereo frame의 자세는 아래와 같다. 
        stframe_curr->getLeft()->setPose(PoseSE3::Identity());
        stframe_curr->getRight()->setPose(PoseSE3::Identity()*stereo_cam_->getRectifiedStereoPoseLeft2Right()); // lower cam 에 대한 자세 .. 슬 필요 없다.

        // Extract initial feature points.
        timer::tic();
        StereoLandmarkTracking lmtrack_curr;
        extractor_->resetWeightBin();
        extractor_->extractORBwithBinning_fast(I1_left, lmtrack_curr.pts_l1, true);
        lmtrack_curr.n_pts = lmtrack_curr.pts_l1.size();
        std::cout << "# extracted features : " << lmtrack_curr.n_pts << std::endl;
		std::cout << colorcode::text_green << "Time [extract ]: " << timer::toc(0) << " [ms]\n" << colorcode::cout_reset;       

        // make dummy pixels
        lmtrack_curr.pts_l0.resize(lmtrack_curr.n_pts);
        lmtrack_curr.pts_r0.resize(lmtrack_curr.n_pts);
        lmtrack_curr.pts_r1.resize(lmtrack_curr.n_pts);
        lmtrack_curr.lms.resize(lmtrack_curr.n_pts);

        // Feature tracking.
        // 1) Track 'static stereo' (I1_left --> I1_right)
        timer::tic();
        MaskVec mask_track(lmtrack_curr.n_pts, true);
        this->tracker_->trackBidirection(I1_left, I1_right,
            lmtrack_curr.pts_l1, params_.feature_tracker.window_size, params_.feature_tracker.max_level, params_.feature_tracker.thres_error, params_.feature_tracker.thres_bidirection,
            lmtrack_curr.pts_r1, mask_track); // lmtrack_curr.pts_u1 에 prior pixels가 이미 들어있다.

        StereoLandmarkTracking lmtrack_staticklt;
        for(int i = 0; i < mask_track.size(); ++i)
        {
            if( mask_track[i] )
            {
                lmtrack_staticklt.pts_l1.push_back(lmtrack_curr.pts_l1[i]);
                lmtrack_staticklt.pts_r1.push_back(lmtrack_curr.pts_r1[i]);
            }
        }
        lmtrack_staticklt.n_pts = lmtrack_staticklt.pts_l1.size();
        lmtrack_staticklt.pts_l0.resize(lmtrack_staticklt.n_pts);
        lmtrack_staticklt.pts_r1.resize(lmtrack_staticklt.n_pts);
        lmtrack_staticklt.lms.resize(lmtrack_staticklt.n_pts);

		std::cout << colorcode::text_green << "Time [track bidirection]: " << timer::toc(0) << " [ms]\n" << colorcode::cout_reset;       

        // make landmarks
        for(int i = 0; i < lmtrack_staticklt.n_pts; ++i)
        {
            const Pixel& pt_left  = lmtrack_staticklt.pts_l1[i];
            const Pixel& pt_right = lmtrack_staticklt.pts_r1[i];
            LandmarkPtr lmptr = std::make_shared<Landmark>(pt_left, stframe_curr->getLeft(), cam_rect);
            lmptr->addObservationAndRelatedFrame(pt_right, stframe_curr->getRight());

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
            const Pixel&      pt0 = lmtrack_staticklt.pts_l1[i];
            const Pixel&      pt1 = lmtrack_staticklt.pts_r1[i];
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
        stframe_curr->getLeft()->setPtsSeenAndRelatedLandmarks(lmtrack_staticklt.pts_l1, lmtrack_staticklt.lms);
        stframe_curr->getRight()->setPtsSeenAndRelatedLandmarks(lmtrack_staticklt.pts_r1, lmtrack_staticklt.lms);


        this->system_flags_.flagFirstImageGot = true;

        ROS_INFO_STREAM("============ End initialization. Start to iterate all images... ============");
        

        if(0) 
        {
            cv::Mat img_color;
            I1_right.convertTo(img_color, CV_8UC1);
            cv::cvtColor(img_color, img_color, CV_GRAY2RGB);
            for(const auto& pt : lmtrack_staticklt.pts_r1)
                cv::circle(img_color, pt, 5, cv::Scalar(0,255,0));
            
            std::stringstream ss;
            ss << " - feature_backtrack";
            cv::namedWindow(ss.str());
            cv::imshow(ss.str(), img_color);


            I1_left.convertTo(img_color, CV_8UC1);
            cv::cvtColor(img_color, img_color, CV_GRAY2RGB);
            for(const auto& pt : lmtrack_staticklt.pts_l1)
                cv::circle(img_color, pt, 5, cv::Scalar(0,255,0));
            
            ss;
            ss << " - feature_track";
            cv::namedWindow(ss.str());
            cv::imshow(ss.str(), img_color);
            cv::waitKey(0);
        }

    }

// [6] Update prev
	this->stframe_prev_ = stframe_curr;	

	std::cout << "# of all landmarks: " << all_landmarks_.size() << std::endl;

};
