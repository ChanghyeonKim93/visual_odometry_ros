#include "core/scale_mono_vo.h"

/**
 * @brief function to track a new image (local bundle mode)
 * @details 새로 들어온 이미지의 자세를 구하는 함수. 만약, scale mono vo가 초기화되지 않은 경우, 해당 이미지를 초기 이미지로 설정. 
 * @param img 입력 이미지 (CV_8UC1)
 * @param timestamp 입력 이미지의 timestamp.
 * @return none
 * @author Changhyeon Kim (hyun91015@gmail.com)
 * @date 10-July-2022
 */
void ScaleMonoVO::trackImageLocalBundle2(const cv::Mat& img, const double& timestamp){
	float THRES_SAMPSON  = params_.feature_tracker.thres_sampson;
	float THRES_PARALLAX = params_.map_update.thres_parallax;

	// Generate statistics
	AlgorithmStatistics::LandmarkStatistics  statcurr_landmark;
	AlgorithmStatistics::FrameStatistics     statcurr_frame;
	AlgorithmStatistics::ExecutionStatistics statcurr_execution;
			
	// 현재 이미지에 대한 새로운 Frame 생성
	FramePtr frame_curr = std::make_shared<Frame>();
	this->saveFrames(frame_curr);
	
	// 이미지 undistort (KITTI라서 할 필요 X)
	cv::Mat img_undist;
	if(system_flags_.flagDoUndistortion) 
	{
		cam_->undistortImage(img, img_undist);
		img_undist.convertTo(img_undist, CV_8UC1);
	}
	else img.copyTo(img_undist);

	// frame_curr에 img_undist와 시간 부여 (gradient image도 함께 사용)
	frame_curr->setImageAndTimestamp(img_undist, timestamp);

	// Get previous and current images
	const cv::Mat& I0 = frame_prev_->getImage();
	const cv::Mat& I1 = frame_curr->getImage();

	if( !system_flags_.flagVOInit ) 
	{ 
		// 초기화 미완료
		if( !system_flags_.flagFirstImageGot ) 
		{ 
			// 최초 이미지	
			// Extract pixels
			LandmarkTracking lmtrack_curr;
			extractor_->resetWeightBin();
			extractor_->extractORBwithBinning(I1, lmtrack_curr.pts1, true);
			std::cout << "# features : " << lmtrack_curr.pts1.size()  << std::endl;

			// 초기 landmark 생성
			for(auto pt : lmtrack_curr.pts1)
			{
				LandmarkPtr lm_new = std::make_shared<Landmark>(pt, frame_curr);
				lmtrack_curr.lms.push_back(lm_new);
			}
			
			// Related Landmark와 tracked pixels를 업데이트
			frame_curr->setPtsSeenAndRelatedLandmarks(lmtrack_curr.pts1, lmtrack_curr.lms);

			frame_curr->setPose(PoseSE3::Identity());
			PoseSE3 T_init = PoseSE3::Identity();
			T_init.block<3,1>(0,3) << 0,0,-0.85; // get initial scale.
			frame_curr->setPoseDiff10(T_init);
			
			this->saveLandmarks(lmtrack_curr.lms); // save all newly detected landmarks

			if( true )
				this->showTracking("img_features", I1, lmtrack_curr.pts1, PixelVec(), PixelVec());

			// 첫 이미지 업데이트 완료
			system_flags_.flagFirstImageGot = true;
		}
		else 
		{
			LandmarkTracking lmtrack_prev;
			lmtrack_prev.pts0 = frame_prev_->getPtsSeen();
			lmtrack_prev.pts1 = PixelVec();
			lmtrack_prev.lms  = frame_prev_->getRelatedLandmarkPtr();

			// 이전 자세의 변화량을 가져온다. 
			PoseSE3 Twc_prev   = frame_prev_->getPose();
			PoseSE3 dT01_prior = frame_prev_->getPoseDiff01();

			// frame_prev_ 의 lms 를 현재 이미지로 track. 5ms
			MaskVec  mask_track;
			tracker_->track(I0, I1, lmtrack_prev.pts0, params_.feature_tracker.window_size, params_.feature_tracker.max_level, params_.feature_tracker.thres_error,
				lmtrack_prev.pts1, mask_track);

			LandmarkTracking lmtrack_klt;
			this->pruneInvalidLandmarks(lmtrack_prev, mask_track, lmtrack_klt);

			// Scale refinement 50ms
			MaskVec mask_refine(lmtrack_klt.pts0.size(), true);			
			LandmarkTracking lmtrack_scaleok;
			this->pruneInvalidLandmarks(lmtrack_klt, mask_refine, lmtrack_scaleok);

			// 5-point algorithm 2ms
			MaskVec mask_5p(lmtrack_scaleok.pts0.size());
			PointVec X0_inlier(lmtrack_scaleok.pts0.size());
			Rot3 dR10; Pos3 dt10;
			if( !motion_estimator_->calcPose5PointsAlgorithm(lmtrack_scaleok.pts0, lmtrack_scaleok.pts1, cam_, dR10, dt10, X0_inlier, mask_5p) ) 
				throw std::runtime_error("calcPose5PointsAlgorithm() is failed.");

			// Check sampson distance 0.01 ms
			std::vector<float> symm_epi_dist;
			motion_estimator_->calcSampsonDistance(lmtrack_scaleok.pts0, lmtrack_scaleok.pts1, cam_, dR10, dt10, symm_epi_dist);
			MaskVec mask_sampson(lmtrack_scaleok.pts0.size());
			for(int i = 0; i < mask_sampson.size(); ++i)
				mask_sampson[i] = mask_5p[i] && (symm_epi_dist[i] < THRES_SAMPSON);
			
			LandmarkTracking lmtrack_final;
			this->pruneInvalidLandmarks(lmtrack_scaleok, mask_sampson, lmtrack_final);

			// Update tracking results
			for(int i = 0; i < lmtrack_final.lms.size(); ++i)
				lmtrack_final.lms[i]->addObservationAndRelatedFrame(lmtrack_final.pts1[i], frame_curr);
			
			// Frame_curr의 자세를 넣는다.
			dt10 = dt10/dt10.norm()*params_.scale_estimator.initial_scale;
			PoseSE3 dT10; dT10 << dR10, dt10, 0.0f, 0.0f, 0.0f, 1.0f;
			PoseSE3 dT01 = geometry::inverseSE3_f(dT10);

			frame_curr->setPose(Twc_prev*dT01);
			frame_curr->setPoseDiff10(dT10);
							
#ifdef RECORD_FRAME_STAT
statcurr_frame.Twc   = frame_curr->getPose();
statcurr_frame.Tcw   = frame_curr->getPoseInv();
statcurr_frame.dT_10 = frame_curr->getPoseDiff10();
statcurr_frame.dT_01 = frame_curr->getPoseDiff01();
#endif
			
			// 빈 곳에 특징점 pts1_new 를 추출한다. 2 ms
			PixelVec pts1_new;
			extractor_->updateWeightBin(lmtrack_final.pts1); // 이미 pts1가 있는 곳은 제외.
			extractor_->extractORBwithBinning(I1, pts1_new, true);

			if( true )
				this->showTracking("img_features", I1, lmtrack_final.pts0, lmtrack_final.pts1, pts1_new);
			
			if( pts1_new.size() > 0 )
			{
				// 새로운 특징점을 back-track.
				PixelVec pts0_new;
				MaskVec mask_new;
				tracker_->trackBidirection(I1, I0, pts1_new, params_.feature_tracker.window_size, params_.feature_tracker.max_level, params_.feature_tracker.thres_error, params_.feature_tracker.thres_bidirection,
					pts0_new, mask_new);

				// 새로운 특징점은 새로운 landmark가 된다.
				for(int i = 0; i < pts1_new.size(); ++i) 
				{
					if( mask_new[i] )
					{
						const Pixel& p0_new = pts0_new[i];
						const Pixel& p1_new = pts1_new[i];
						LandmarkPtr ptr = std::make_shared<Landmark>(p0_new, frame_prev_);
						ptr->addObservationAndRelatedFrame(p1_new, frame_curr);

						lmtrack_final.pts0.push_back(p0_new);
						lmtrack_final.pts1.push_back(p1_new);
						lmtrack_final.lms.push_back(ptr);
						this->saveLandmarks(ptr);
					}
				}
			}

			// lms1_final 중, depth가 복원되지 않은 경우 복원해준다.
			uint32_t cnt_recon = 0 ;
			for(auto lm : lmtrack_final.lms)
			{
				if( !lm->isTriangulated() && lm->getLastParallax() >= THRES_PARALLAX)
				{
					if(lm->getObservations().size() != lm->getRelatedFramePtr().size())
						throw std::runtime_error("lm->getObservations().size() != lm->getRelatedFramePtr().size()\n");

					const Pixel& pt0 = lm->getObservations().front();
					const Pixel& pt1 = lm->getObservations().back();
					const PoseSE3& Tw0 = lm->getRelatedFramePtr().front()->getPose();
					const PoseSE3& Tw1 = lm->getRelatedFramePtr().back()->getPose();
					PoseSE3 T10 =  Tw1.inverse() * Tw0;

					// Reconstruct points
					Point X0, X1;
					Mapping::triangulateDLT(pt0, pt1, T10.block<3,3>(0,0), T10.block<3,1>(0,3), cam_, X0, X1);

					if(X0(2) > 0)
					{
						Point Xworld = Tw0.block<3,3>(0,0)*X0 + Tw0.block<3,1>(0,3);
						lm->set3DPoint(Xworld);

						++cnt_recon;
					}
				}
			}
			std::cout << " Recon done. : " << cnt_recon << "\n";

			// lms1와 pts1을 frame_curr에 넣는다.
			frame_curr->setPtsSeenAndRelatedLandmarks(lmtrack_final.pts1, lmtrack_final.lms);

			system_flags_.flagVOInit = true;
		}
	}
	else 
	{
		// VO initialized. Do track the new image.
		LandmarkTracking lmtrack_prev;
		lmtrack_prev.pts0 = frame_prev_->getPtsSeen();
		lmtrack_prev.pts1 = PixelVec();
		lmtrack_prev.lms  = frame_prev_->getRelatedLandmarkPtr();

		// 이전 자세의 변화량을 가져온다. 
		PoseSE3 Twc_prev   = frame_prev_->getPose();
		PoseSE3 Tcw_prev   = geometry::inverseSE3_f(Twc_prev);

		PoseSE3 dT01_prior = frame_prev_->getPoseDiff01();
		PoseSE3 Twc_prior  = Twc_prev*dT01_prior;
		PoseSE3 Tcw_prior  = geometry::inverseSE3_f(Twc_prior);

		// Make tracking prior & estimated scale
		lmtrack_prev.pts1.resize(lmtrack_prev.pts0.size());
		for(int i = 0; i < lmtrack_prev.pts0.size(); ++i){
			const LandmarkPtr& lm = lmtrack_prev.lms[i];
			float patch_scale = 1.0f;
			if( lm->isBundled() ){
				const Point& Xw = lm->get3DPoint();
				Point Xp = Tcw_prev.block<3,3>(0,0)*Xw + Tcw_prev.block<3,1>(0,3);
				Point Xc = Tcw_prior.block<3,3>(0,0)*Xw + Tcw_prior.block<3,1>(0,3);
				float dp = Xp(2); float dc = Xc(2);

				patch_scale = dp/dc;
				// std::cout << "scale : " << patch_scale << std::endl;
				
				if(Xc(2) > 0) lmtrack_prev.pts1[i] = cam_->projectToPixel(Xc);
				else lmtrack_prev.pts1[i] = lmtrack_prev.pts0[i];
			}
			else lmtrack_prev.pts1[i] = lmtrack_prev.pts0[i];

			lmtrack_prev.scale_change.push_back(patch_scale);
		}

		// frame_prev_ 의 lms 를 현재 이미지로 track. 5ms
		MaskVec  mask_track;
		// tracker_->trackWithPrior(I0, I1, lmtrack_prev.pts0, params_.feature_tracker.window_size, params_.feature_tracker.max_level, params_.feature_tracker.thres_error,
			// lmtrack_prev.pts1, mask_track);
		tracker_->trackBidirectionWithPrior(I0, I1, lmtrack_prev.pts0, params_.feature_tracker.window_size, params_.feature_tracker.max_level, params_.feature_tracker.thres_error, params_.feature_tracker.thres_bidirection,
			lmtrack_prev.pts1, mask_track);

		LandmarkTracking lmtrack_kltok;
		std::cout << "# of PREV  : " << lmtrack_prev.pts0.size() << std::endl;
		std::cout << "# of kltok : " << this->pruneInvalidLandmarks(lmtrack_prev, mask_track, lmtrack_kltok) << std::endl;

		// Scale refinement 50ms
		timer::tic();
		MaskVec mask_refine(lmtrack_kltok.pts0.size(), true);
		const cv::Mat& du0 = frame_prev_->getImageDu();
		const cv::Mat& dv0 = frame_prev_->getImageDv();
		tracker_->trackWithScale(
			I0, du0, dv0, I1, 
			lmtrack_kltok.pts0, lmtrack_kltok.scale_change, lmtrack_kltok.pts1,
			mask_refine);
		std::cout << "trackWithScale time : " << timer::toc(0) << " [ms]\n";

		LandmarkTracking lmtrack_scaleok;
		std::cout << "# of scaleok : " << this->pruneInvalidLandmarks(lmtrack_kltok, mask_refine, lmtrack_scaleok) << std::endl;
		
		// 깊이를 가진 점 갯수를 세어보고, 30개 이상이면 local bundle을 수행한다.
		uint32_t cnt_depth_ok = 0;
		PixelVec pts1_depth_ok; 
		PointVec Xp_depth_ok; 
		pts1_depth_ok.reserve(lmtrack_scaleok.pts0.size());
		Xp_depth_ok.reserve(lmtrack_scaleok.pts0.size());

		Rot3 Rcw_prev = Tcw_prev.block<3,3>(0,0);
		Pos3 tcw_prev = Tcw_prev.block<3,1>(0,3);
		LandmarkPtrVec lms1_depthok;
		PixelVec pts1_project;
		if(keyframes_->getList().size() > 5){ // # of keyframes is over 5
			for(int i = 0; i < lmtrack_scaleok.pts0.size(); ++i){
				const LandmarkPtr& lm = lmtrack_scaleok.lms[i];
				if( lm->isBundled() ) {
					Point Xp = Rcw_prev * lm->get3DPoint() + tcw_prev;
					if(Xp(2) > 0) {
						pts1_depth_ok.push_back(lmtrack_scaleok.pts1[i]);
						Xp_depth_ok.push_back(Xp);
						++cnt_depth_ok;
					}
				}
			}
		}
		else{
			for(int i = 0; i < lmtrack_scaleok.pts0.size(); ++i){
				const LandmarkPtr& lm = lmtrack_scaleok.lms[i];
				if(lm->isTriangulated() && lm->getAge() > 1 
				&& lm->getLastParallax() > THRES_PARALLAX){ 
					Point Xp = Rcw_prev * lm->get3DPoint() + tcw_prev;
					if(Xp(2) > 0){
						pts1_depth_ok.push_back(lmtrack_scaleok.pts1[i]);
						Xp_depth_ok.push_back(Xp);
						++cnt_depth_ok;
					}
				}
			}
		}
		
		pts1_project = pts1_depth_ok;
		
		MaskVec mask_motion(lmtrack_scaleok.pts0.size(), true);
		Rot3 dR10; Pos3 dt10; PoseSE3 dT10;
		Rot3 dR01; Pos3 dt01; PoseSE3 dT01;

		bool flag_do_5point = false;
		if(cnt_depth_ok > 10){
			// Do Local BA
			std::cout << " DO pose-only Bundle Adjustment... with [" << cnt_depth_ok <<"] points.\n";

			dR01 = dT01_prior.block<3,3>(0,0);
			dt01 = dT01_prior.block<3,1>(0,3);
			std::cout <<"======== prior dt01: " << dt01.transpose() <<std::endl;

			timer::tic();
			if(motion_estimator_->calcPoseOnlyBundleAdjustment(Xp_depth_ok, pts1_depth_ok, cam_, params_.motion_estimator.thres_poseba_error,
					dR01, dt01, mask_motion))
			{
				dT01 << dR01, dt01, 0,0,0,1;
				dT10 = geometry::inverseSE3_f(dT01);

				dR10 = dT10.block<3,3>(0,0);
				dt10 = dT10.block<3,1>(0,3);

				if(std::isnan(dt10.norm())){
					throw std::runtime_error("std::isnan(dt01.norm()) ...");
					flag_do_5point = true;
				}
				else {
					frame_curr->setPose(Twc_prev*dT01);		
					frame_curr->setPoseDiff10(dT10);	
				}

				// Projection 
				for(int i = 0; i < Xp_depth_ok.size(); ++i){
					Point Xc = dR10*Xp_depth_ok[i] + dt10;
					pts1_project[i] = cam_->projectToPixel(Xc);
				}
				std::cout <<"======== est   dt01: " << dt01.transpose() <<std::endl;

			}
			else flag_do_5point = true; // Failed to converge. Re-estimate motion with 5-point algorithm.
			timer::toc(1);
		}
		else flag_do_5point = true; // Not enough points. Estimate motion with 5-point algorithm.

		if(flag_do_5point) { // do 5 point algorihtm (scale is of the previous frame)
			std::cout << "\n\n\n!!!!!!!!!!!!!!!!!!!!!! -- WARNING ! DO 5-points algorithm -- !!!!!!!!!!!!!!!!!!!!! \n\n\n\n";
			PointVec X0_inlier(lmtrack_scaleok.pts0.size());
		
			if( !motion_estimator_->calcPose5PointsAlgorithm(lmtrack_scaleok.pts0, lmtrack_scaleok.pts1, cam_, dR10, dt10, X0_inlier, mask_motion) ) 
				throw std::runtime_error("calcPose5PointsAlgorithm() is failed.");

			// Frame_curr의 자세를 넣는다.
			float scale = frame_prev_->getPoseDiff01().block<3,1>(0,3).norm();
			dT10 << dR10, (scale/dt10.norm())*dt10, 0.0f, 0.0f, 0.0f, 1.0f;
			dT01 = geometry::inverseSE3_f(dT10);

			frame_curr->setPose(Twc_prev*dT01);		
			frame_curr->setPoseDiff10(dT10);	
		}
		LandmarkTracking lmtrack_motion;
		std::cout << "# of motion: " << this->pruneInvalidLandmarks(lmtrack_scaleok, mask_motion, lmtrack_motion) << std::endl;

		// Check sampson distance 0.01 ms
		std::vector<float> symm_epi_dist;
		motion_estimator_->calcSampsonDistance(lmtrack_motion.pts0, lmtrack_motion.pts1, cam_, dT10.block<3,3>(0,0), dT10.block<3,1>(0,3), symm_epi_dist);
		MaskVec mask_sampson(lmtrack_motion.pts0.size(),true);
		for(int i = 0; i < mask_sampson.size(); ++i)
			mask_sampson[i] = symm_epi_dist[i] < params_.feature_tracker.thres_sampson;
		
		LandmarkTracking lmtrack_final;
		std::cout << "# of samps : " << this->pruneInvalidLandmarks(lmtrack_motion, mask_sampson, lmtrack_final) << std::endl;

		for(int i = 0; i < lmtrack_final.pts0.size(); ++i)
			lmtrack_final.lms[i]->addObservationAndRelatedFrame(lmtrack_final.pts1[i], frame_curr);
				
#ifdef RECORD_FRAME_STAT
statcurr_frame.Twc   = frame_curr->getPose();
statcurr_frame.Tcw   = frame_curr->getPoseInv();
statcurr_frame.dT_10 = frame_curr->getPoseDiff10();
statcurr_frame.dT_01 = frame_curr->getPoseDiff01();
#endif
			
		// 빈 곳에 특징점 pts1_new 를 추출한다. 2 ms
		PixelVec pts1_new;
		extractor_->updateWeightBin(lmtrack_final.pts1); // 이미 pts1가 있는 곳은 제외.
		extractor_->extractORBwithBinning(I1, pts1_new, true);
		std::cout << "# features : " << pts1_new.size()  << std::endl;

		if( true )
			this->showTrackingBA("img_feautues", I1, pts1_depth_ok, pts1_project, mask_sampson);
			// this->showTracking("img_features", I1, lmtrack_final.pts0, lmtrack_final.pts1, pts1_new);
		
		if( pts1_new.size() > 0 ){
			// 새로운 특징점을 back-track.
			PixelVec pts0_new;
			MaskVec mask_new;
			tracker_->trackBidirection(I1, I0, pts1_new, params_.feature_tracker.window_size, params_.feature_tracker.max_level, params_.feature_tracker.thres_error, params_.feature_tracker.thres_bidirection,
				pts0_new, mask_new);

			// 새로운 특징점은 새로운 landmark가 된다.
			for(int i = 0; i < pts1_new.size(); ++i) {
				if( mask_new[i] ){
					const Pixel& p0_new = pts0_new[i];
					const Pixel& p1_new = pts1_new[i];
					LandmarkPtr ptr = std::make_shared<Landmark>(p0_new, frame_prev_);
					ptr->addObservationAndRelatedFrame(p1_new, frame_curr);

					lmtrack_final.pts0.push_back(p0_new);
					lmtrack_final.pts1.push_back(p1_new);
					lmtrack_final.lms.push_back(ptr);
					this->saveLandmarks(ptr);
				}
			}
		}

		// lms1와 pts1을 frame_curr에 넣는다.
		frame_curr->setPtsSeenAndRelatedLandmarks(lmtrack_final.pts1, lmtrack_final.lms);
	}

	// Check keyframe update rules.
	if(keyframes_->checkUpdateRule(frame_curr))
	{
		// If there exist keyframe already, check turning region or not.
		if(keyframes_->getList().size() > 0)
		{
			// Find turning region
			if(scale_estimator_->detectTurnRegions(keyframes_, frame_curr))
			{
				std::cout << " ===  ===  Turn region is detected! \n";

				FramePtrVec Ft = scale_estimator_->getAllTurnRegions(); // all turn regions
				for(auto f : Ft) stat_.stat_turn.turn_regions.push_back(f);

				// VO를 통해 유지되어온 scale 궤적. 
				std::vector<float> scales_vo;
				for(auto f : scale_estimator_->getLastTurnRegion()){
					PoseSE3 dT01_f = f->getPoseDiff01();
					scales_vo.push_back(dT01_f.block<3,1>(0,3).norm());
				}
				std::sort(scales_vo.begin(), scales_vo.end());

				// Find median 
				// int idx_median = (int)((float)scales_vo.size()*0.5);
				// std::cout << scales_vo.size() <<" , " <<  idx_median << std::endl;
				// float scaler = scale_estimator_->getLastTurnRegion().back()->getScale()/scales_vo[idx_median];
				
				// std::cout <<" scaler : " << scaler << std::endl;
				// for(auto f : scale_estimator_->getLastTurnRegion()){
				// 	PoseSE3 dT01_f = f->getPoseDiff01();
				// 	PoseSE3 Twc = f->getPose();
				// 	Twc*= dT01_f.inverse();
				// 	dT01_f.block<3,1>(0,3) *= scaler;
				// 	Twc*= dT01_f;
				// 	f->setPose(Twc);
				// 	f->setPoseDiff10(dT01_f.inverse());
				// }
			}

		}
		

		// Add new keyframe
		keyframes_->addNewKeyframe(frame_curr);

		// Make variables for refine tracking
		const LandmarkPtrVec& lms_final = frame_curr->getRelatedLandmarkPtr();
		FloatVec scale_estimated(lms_final.size(), 1.0f);
		MaskVec mask_final(lms_final.size(), true);
		PixelVec pts_refine(lms_final.size());

		for(int i = 0; i < lms_final.size(); ++i){
			// Estimate current scale...
			const LandmarkPtr& lm = lms_final[i];
			if(lm->isTriangulated()) {
				const Point& Xw = lm->get3DPoint();

				const PoseSE3& T0w = lm->getRelatedFramePtr().front()->getPoseInv();
				const PoseSE3& T1w = lm->getRelatedFramePtr().back()->getPoseInv();

				Point X0  = T0w.block<3,3>(0,0)*Xw + T0w.block<3,1>(0,3);
				Point X1  = T1w.block<3,3>(0,0)*Xw + T1w.block<3,1>(0,3);

				float d0 = X0(2), d1 = X1(2);
				float scale = d0/d1;

				mask_final[i]      = true;
				scale_estimated[i] = scale;
				pts_refine[i]      = lm->getObservations().back();

				// std::cout << i << "-th point:" << Xw.transpose() << " scale: " << scale << std::endl;
			}
			else
				mask_final[i] = false;
		}

		// Refine the tracking results
		timer::tic();
		tracker_->refineTrackWithScale(I1, lms_final, scale_estimated, pts_refine, mask_final);
		std::cout << "trackWithScale time : " << timer::toc(0) << " [ms]\n";

		// Update points
		for(int i = 0; i < lms_final.size(); ++i)
		{
			const LandmarkPtr& lm = lms_final[i];
			if(mask_final[i])
				lm->changeLastObservation(pts_refine[i]);
		}

		// Reconstruct map points
		// lms1_final 중, depth가 복원되지 않은 경우 복원해준다.
		uint32_t cnt_recon = 0;
		
		for(auto lm : frame_curr->getRelatedLandmarkPtr())
		{
			if(!lm->isTriangulated() && lm->getLastParallax() >= THRES_PARALLAX)
			{
				if(lm->getObservationsOnKeyframes().size() > 1)
				{
					// 2번 이상 keyframe에서 보였다.
					const Pixel& pt0 = lm->getObservationsOnKeyframes().front();
					const Pixel& pt1 = lm->getObservationsOnKeyframes().back();

					const PoseSE3& Tw0 = lm->getRelatedKeyframePtr().front()->getPose();
					const PoseSE3& Tw1 = lm->getRelatedKeyframePtr().back()->getPose();
					PoseSE3 T10_tmp = geometry::inverseSE3_f(Tw1) * Tw0;

					// Reconstruct points
					Point X0, X1;
					Mapping::triangulateDLT(pt0, pt1, T10_tmp.block<3,3>(0,0), T10_tmp.block<3,1>(0,3), cam_, X0, X1);

					// Check reprojection error for the first image
					Pixel pt0_proj = cam_->projectToPixel(X0);
					Pixel dpt0 = pt0 - pt0_proj;
					float dpt0_norm2 = dpt0.x*dpt0.x + dpt0.y*dpt0.y;
					
					if(dpt0_norm2 > 1.0) 
						continue;

					Pixel pt1_proj = cam_->projectToPixel(X1);
					Pixel dpt1 = pt1 - pt1_proj;
					float dpt1_norm2 = dpt1.x*dpt1.x + dpt1.y*dpt1.y;
					
					if(dpt1_norm2 > 1.0) 
						continue;

					// Check the point in front of cameras
					if(X0(2) > 0 && X1(2) > 0)
					{
						Point Xworld = Tw0.block<3,3>(0,0)*X0 + Tw0.block<3,1>(0,3);
						lm->set3DPoint(Xworld);
						++cnt_recon;
					}
				}
			}
		}
		std::cout << " Recon done. : " << cnt_recon << "\n";

		// Do local bundle adjustment for keyframes.
		// motion_estimator_->localBundleAdjustment(keyframes_, cam_);
		motion_estimator_->localBundleAdjustmentSparseSolver(keyframes_, cam_);
	}
	
	// Replace the 'frame_prev_' with 'frame_curr'
	frame_prev_ = frame_curr;

	// Visualization 3D points
	PointVec X_world_recon;
	X_world_recon.reserve(all_landmarks_.size());
	for(auto lm : all_landmarks_){
		if(lm->isTriangulated()) 
			X_world_recon.push_back(lm->get3DPoint());
	}
	std::cout << "# of all landmarks: " << X_world_recon.size() << std::endl;

#ifdef RECORD_FRAME_STAT
statcurr_frame.mappoints.resize(0);
statcurr_frame.mappoints = X_world_recon;
#endif

	// Update statistics
	stat_.stats_landmark.push_back(statcurr_landmark);
	stat_.stats_frame.resize(0);
	stat_.stats_frame.push_back(statcurr_frame);
	stat_.stats_execution.push_back(statcurr_execution);
	std::cout << "Statistics Updated. size: " << stat_.stats_landmark.size() << "\n";

	// Notify a thread.
	mut_scale_estimator_->lock();
	*flag_do_ASR_ = true;
	mut_scale_estimator_->unlock();
	cond_var_scale_estimator_->notify_all();
};