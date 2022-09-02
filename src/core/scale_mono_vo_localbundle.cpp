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
void ScaleMonoVO::trackImageLocalBundle(const cv::Mat& img, const double& timestamp){
	
	float THRES_ZNCC    = 0.90f;
	float THRES_SAMPSON = 2.0f;

	// Generate statistics
	AlgorithmStatistics::LandmarkStatistics  statcurr_landmark;
	AlgorithmStatistics::FrameStatistics     statcurr_frame;
	AlgorithmStatistics::ExecutionStatistics statcurr_execution;
			
	// 현재 이미지에 대한 새로운 Frame 생성
	FramePtr frame_curr = std::make_shared<Frame>();
	this->saveFrames(frame_curr);
	
	// 이미지 undistort (KITTI라서 할 필요 X)
	cv::Mat img_undist;
	if(system_flags_.flagDoUndistortion) {
		cam_->undistortImage(img, img_undist);
		img_undist.convertTo(img_undist, CV_8UC1);
	}
	else img.copyTo(img_undist);

	// frame_curr에 img_undist와 시간 부여
	frame_curr->setImageAndTimestamp(img_undist, timestamp);

	// Get previous and current images
	const cv::Mat& I0 = frame_prev_->getImage();
	const cv::Mat& I1 = frame_curr->getImage();

	if( !system_flags_.flagVOInit ) { // 초기화 미완료
		if( !system_flags_.flagFirstImageGot ) { // 최초 이미지	
			// Extract pixels
			PixelVec       pts1;
			LandmarkPtrVec lms1;
			extractor_->resetWeightBin();
			extractor_->extractORBwithBinning(I1, pts1);

			// 초기 landmark 생성
			lms1.reserve(pts1.size());
			for(auto p : pts1) lms1.push_back(std::make_shared<Landmark>(p, frame_curr));
			
			// Related Landmark와 tracked pixels를 업데이트
			frame_curr->setPtsSeenAndRelatedLandmarks(pts1,lms1);

			frame_curr->setPose(PoseSE3::Identity());
			PoseSE3 T_init = PoseSE3::Identity();
			T_init.block<3,1>(0,3) << 0,0,-0.90;
			frame_curr->setPoseDiff10(T_init);
			
			this->saveLandmarks(lms1);	

			if( true )
				this->showTracking("img_features", I1, pts1, PixelVec(), PixelVec());

			// 첫 이미지 업데이트 완료
			system_flags_.flagFirstImageGot = true;
		}
		else {
			// 초기화 미완료
			const PixelVec&       pts0 = frame_prev_->getPtsSeen();
			const LandmarkPtrVec& lms0 = frame_prev_->getRelatedLandmarkPtr();

			// 이전 자세의 변화량을 가져온다. 
			PoseSE3 Twc_prev   = frame_prev_->getPose();
			PoseSE3 dT01_prior = frame_prev_->getPoseDiff01();
			PoseSE3 Twc_prior  = Twc_prev*dT01_prior;
			PoseSE3 Tcw_prior  = Twc_prior.inverse();

			// frame_prev_ 의 lms 를 현재 이미지로 track. 5ms
			PixelVec pts1;
			MaskVec  mask_track;
			tracker_->track(I0, I1, pts0, params_.feature_tracker.window_size, params_.feature_tracker.max_level, params_.feature_tracker.thres_error,
				pts1, mask_track);
			PixelVec       pts0_trackok;
			PixelVec       pts1_trackok;
			LandmarkPtrVec lms1_trackok;
			this->pruneInvalidLandmarks(pts0, pts1, lms0, mask_track, 
				pts0_trackok, pts1_trackok, lms1_trackok);

			// Scale refinement 50ms
			MaskVec mask_refine(pts0_trackok.size(), true);
			// tracker_->refineScale(I0, I1, frame_curr->getImageDu(), frame_curr->getImageDv(), pts0_trackok, 1.25f, 
			// 	pts1_trackok, mask_refine);
			
			PixelVec       pts0_scaleok;
			PixelVec       pts1_scaleok;
			LandmarkPtrVec lms1_scaleok;
			this->pruneInvalidLandmarks(pts0_trackok, pts1_trackok, lms1_trackok, mask_refine, 
				pts0_scaleok, pts1_scaleok, lms1_scaleok);

			// 5-point algorithm 2ms

			MaskVec mask_5p(pts0_scaleok.size());
			PointVec X0_inlier(pts0_scaleok.size());
			Rot3 dR10;
			Pos3 dt10;
			if( !motion_estimator_->calcPose5PointsAlgorithm(pts0_scaleok, pts1_scaleok, cam_, dR10, dt10, X0_inlier, mask_5p) ) {
				throw std::runtime_error("calcPose5PointsAlgorithm() is failed.");
			}

			// Check sampson distance 0.01 ms
			std::vector<float> symm_epi_dist;
			motion_estimator_->calcSampsonDistance(pts0_scaleok, pts1_scaleok, cam_, dR10, dt10, symm_epi_dist);
			MaskVec mask_sampson(pts0_scaleok.size());

			for(int i = 0; i < mask_sampson.size(); ++i){
				mask_sampson[i] = mask_5p[i] && (symm_epi_dist[i] < THRES_SAMPSON);
			}

			PixelVec       pts0_final;
			PixelVec       pts1_final;
			LandmarkPtrVec lms1_final;
			this->pruneInvalidLandmarks(pts0_scaleok, pts1_scaleok, lms1_scaleok, mask_sampson, 
				pts0_final, pts1_final, lms1_final);
			
			// Update tracking results
			for(int i = 0; i < lms1_final.size(); ++i){
				lms1_final[i]->addObservationAndRelatedFrame(pts1_final[i], frame_curr);
			}


			// Frame_curr의 자세를 넣는다.
			PoseSE3 dT10; dT10 << dR10, dt10, 0.0f, 0.0f, 0.0f, 1.0f;
			PoseSE3 dT01 = dT10.inverse();

			frame_curr->setPose(Twc_prev*dT01);		
			frame_curr->setPoseDiff10(dT10);	
				
#ifdef RECORD_FRAME_STAT
statcurr_frame.Twc   = frame_curr->getPose();
statcurr_frame.Tcw   = frame_curr->getPose().inverse();
statcurr_frame.dT_10 = frame_curr->getPoseDiff10();
statcurr_frame.dT_01 = frame_curr->getPoseDiff01();
#endif
			
			// 빈 곳에 특징점 pts1_new 를 추출한다. 2 ms
			PixelVec pts1_new;
			extractor_->updateWeightBin(pts1_final); // 이미 pts1가 있는 곳은 제외.
			extractor_->extractORBwithBinning(I1, pts1_new);

			if( true )
				this->showTracking("img_features", I1, pts0_final, pts1_final, pts1_new);
			
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

						pts0_final.push_back(p0_new);
						pts1_final.push_back(p1_new);
						lms1_final.push_back(ptr);
						this->saveLandmarks(ptr);
					}
				}
			}

			// lms1_final 중, depth가 복원되지 않은 경우 복원해준다.
			uint32_t cnt_recon = 0 ;
			for(auto lm : lms1_final){
				if( !lm->isTriangulated() && lm->getMaxParallax() >= 0.3f*D2R){
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

					if(X0(2) > 0){
						Point Xworld = Tw0.block<3,3>(0,0)*X0 + Tw0.block<3,1>(0,3);
						lm->set3DPoint(Xworld);
						++cnt_recon;
					}
				}
			}
			std::cout << " Recon done. : " << cnt_recon << "\n";

			// lms1와 pts1을 frame_curr에 넣는다.
			frame_curr->setPtsSeenAndRelatedLandmarks(pts1_final,lms1_final);

			system_flags_.flagVOInit = true;
		}
	}
	else { // VO initialized. Do track the new image.
			LandmarkTracking lmtrack_prev;
			lmtrack_prev.pts0 = frame_prev_->getPtsSeen();
			lmtrack_prev.pts1 = PixelVec();
			lmtrack_prev.lms  = frame_prev_->getRelatedLandmarkPtr();

			// 이전 자세의 변화량을 가져온다. 
			PoseSE3 Twc_prev   = frame_prev_->getPose();
			PoseSE3 Tcw_prev   = Twc_prev.inverse();

			PoseSE3 dT01_prior = frame_prev_->getPoseDiff01();
			PoseSE3 Twc_prior  = Twc_prev*dT01_prior;
			PoseSE3 Tcw_prior  = Twc_prior.inverse();

			// Make tracking prior 
			lmtrack_prev.pts1.resize(lmtrack_prev.pts0.size());
			for(int i = 0; i < lmtrack_prev.pts0.size(); ++i){
				const LandmarkPtr& lm = lmtrack_prev.lms[i];
				if(lm->isTriangulated() && lm->getMaxParallax() > 0.5*D2R){
					const Point& Xw = lm->get3DPoint();
					Point Xc = Tcw_prior.block<3,3>(0,0)*Xw + Tcw_prior.block<3,1>(0,3);
					if(Xc(2) > 0) lmtrack_prev.pts1[i] = cam_->projectToPixel(Xc);
					else lmtrack_prev.pts1[i] = lmtrack_prev.pts0[i];
				}
				else lmtrack_prev.pts1[i] = lmtrack_prev.pts0[i];
			}

			// frame_prev_ 의 lms 를 현재 이미지로 track. 5ms
			MaskVec  mask_track;
			tracker_->trackWithPrior(I0, I1, lmtrack_prev.pts0, params_.feature_tracker.window_size, params_.feature_tracker.max_level, params_.feature_tracker.thres_error,
				lmtrack_prev.pts1, mask_track);

			LandmarkTracking lmtrack_kltok;
			this->pruneInvalidLandmarks(lmtrack_prev, mask_track, lmtrack_kltok);

			// Scale refinement 50ms
			MaskVec mask_refine(lmtrack_kltok.pts0.size(), true);
			// tracker_->refineScale(I0, I1, frame_curr->getImageDu(), frame_curr->getImageDv(), pts0_trackok, 1.25f, 
			// 	pts1_trackok, mask_refine);
			
			LandmarkTracking lmtrack_scaleok;
			this->pruneInvalidLandmarks(lmtrack_kltok, mask_refine, lmtrack_scaleok);
			
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
			for(int i = 0; i < lmtrack_scaleok.pts0.size(); ++i){
				const LandmarkPtr& lm = lmtrack_scaleok.lms[i];
				if(lm->isTriangulated() && lm->getAge() > 1 && lm->getMaxParallax() > 0.3*D2R){ 
					Point Xp = Rcw_prev * lm->get3DPoint() + tcw_prev;
					if(Xp(2) > 0){
						pts1_depth_ok.push_back(lmtrack_scaleok.pts1[i]);
						Xp_depth_ok.push_back(Xp);
						++cnt_depth_ok;
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
				if(motion_estimator_->calcPoseOnlyBundleAdjustment(Xp_depth_ok, pts1_depth_ok, cam_, dR01, dt01, mask_motion)){
					dT01 << dR01, dt01, 0,0,0,1;
					dT10 = dT01.inverse();
					dR10 = dT10.block<3,3>(0,0);
					dt10 = dT10.block<3,1>(0,3);
					if(!std::isnan(dt01.norm())){
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
				dT01 = dT10.inverse();

				frame_curr->setPose(Twc_prev*dT01);		
				frame_curr->setPoseDiff10(dT10);	
			}

			// Check sampson distance 0.01 ms
			std::vector<float> symm_epi_dist;
			motion_estimator_->calcSampsonDistance(lmtrack_scaleok.pts0, lmtrack_scaleok.pts1, cam_, dT10.block<3,3>(0,0), dT10.block<3,1>(0,3), symm_epi_dist);
			MaskVec mask_sampson(lmtrack_scaleok.pts0.size(),true);
			for(int i = 0; i < mask_sampson.size(); ++i)
				mask_sampson[i] = mask_motion[i] && (symm_epi_dist[i] < THRES_SAMPSON);
			
			LandmarkTracking lmtrack_final;
			this->pruneInvalidLandmarks(lmtrack_scaleok, mask_sampson, lmtrack_final);

			for(int i = 0; i < lmtrack_final.pts0.size(); ++i)
				lmtrack_final.lms[i]->addObservationAndRelatedFrame(lmtrack_final.pts1[i], frame_curr);
				
#ifdef RECORD_FRAME_STAT
statcurr_frame.Twc = frame_curr->getPose();
statcurr_frame.Tcw = frame_curr->getPose().inverse();
statcurr_frame.dT_10 = frame_curr->getPoseDiff10();
statcurr_frame.dT_01 = frame_curr->getPoseDiff01();
#endif
			
			// 빈 곳에 특징점 pts1_new 를 추출한다. 2 ms
			PixelVec pts1_new;
			extractor_->updateWeightBin(lmtrack_final.pts1); // 이미 pts1가 있는 곳은 제외.
			extractor_->extractORBwithBinning(I1, pts1_new);

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

			// lms1_final 중, depth가 복원되지 않은 경우 복원해준다.
			uint32_t cnt_recon = 0 ;
			for(auto lm : lmtrack_final.lms){
				if(lm->getLastParallax() >= 0.3f*D2R){
					if(lm->getObservations().size() != lm->getRelatedFramePtr().size())
						throw std::runtime_error("lm->getObservations().size() != lm->getRelatedFramePtr().size()\n");

					const Pixel& pt0 = lm->getObservations().front();
					// const Pixel& pt0 = *(lm->getObservations().end()-2);
					const Pixel& pt1 = lm->getObservations().back();
					const PoseSE3& Tw0 = lm->getRelatedFramePtr().front()->getPose();
					// const PoseSE3& Tw0 = (*(lm->getRelatedFramePtr().end()-2))->getPose();
					const PoseSE3& Tw1 = lm->getRelatedFramePtr().back()->getPose();
					PoseSE3 T10 =  Tw1.inverse() * Tw0;

					// Reconstruct points
					Point X0, X1;
					Mapping::triangulateDLT(pt0, pt1, T10.block<3,3>(0,0), T10.block<3,1>(0,3), cam_, X0, X1);

					if(X0(2) > 0){
						Point Xworld = Tw0.block<3,3>(0,0)*X0 + Tw0.block<3,1>(0,3);
						lm->set3DPoint(Xworld);
						++cnt_recon;
					}
				}
			}
			std::cout << " Recon done. : " << cnt_recon << "\n";

			// lms1와 pts1을 frame_curr에 넣는다.
			frame_curr->setPtsSeenAndRelatedLandmarks(lmtrack_final.pts1, lmtrack_final.lms);
	}



	// Update statistics
	stat_.stats_landmark.push_back(statcurr_landmark);
	stat_.stats_frame.push_back(statcurr_frame);
	stat_.stats_execution.push_back(statcurr_execution);
	std::cout << "Statistics Updated. size: " << stat_.stats_landmark.size() << "\n";

	// Check keyframe update rules.
	if(keyframes_->checkUpdateRule(frame_curr)){
		keyframes_->addNewKeyframe(frame_curr);
		
		// Do local bundle adjustment for keyframes.
		// motion_estimator_->localBundleAdjustment(keyframes_, cam_);
		// motion_estimator_->localBundleAdjustmentSparseSolver(keyframes_, cam_);
	}
	
	// Replace the 'frame_prev_' with 'frame_curr'
	frame_prev_ = frame_curr;

	// Visualization 3D points
	PointVec X_world_recon;
	X_world_recon.reserve(all_landmarks_.size());
	for(auto lm : all_landmarks_){
		if(lm->isTriangulated()) {
			X_world_recon.push_back(lm->get3DPoint());
		}
	}
	std::cout << "# of all landmarks: " << X_world_recon.size() << std::endl;

#ifdef RECORD_FRAME_STAT
statcurr_frame.mappoints = X_world_recon;
#endif

	// Notify a thread.
	mut_scale_estimator_->lock();
	*flag_do_ASR_ = true;
	mut_scale_estimator_->unlock();
	cond_var_scale_estimator_->notify_all();
};