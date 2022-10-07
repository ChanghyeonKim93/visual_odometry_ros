#include "core/scale_mono_vo.h"

/**
 * @brief function to track a new image (backend only)
 * @details 새로 들어온 이미지의 자세를 구하는 함수. 만약, scale mono vo가 초기화되지 않은 경우, 해당 이미지를 초기 이미지로 설정. 
 * @param img 입력 이미지 (CV_8UC1)
 * @param timestamp 입력 이미지의 timestamp.
 * @return none
 * @author Changhyeon Kim (hyun91015@gmail.com)
 * @date 10-July-2022
 */
void ScaleMonoVO::trackImageBackend(const cv::Mat& img, const double& timestamp, const PoseSE3& pose, const PoseSE3& dT01) {

	float THRES_ZNCC    = 0.95f;
	float THRES_SAMPSON = 2.0f;
	
	// Generate statistics
	AlgorithmStatistics::LandmarkStatistics  statcurr_landmark;
	AlgorithmStatistics::FrameStatistics     statcurr_frame;
	AlgorithmStatistics::ExecutionStatistics statcurr_execution;

	std::cout << "start track backend...\n";

	// 현재 이미지에 대한 새로운 Frame 생성
	FramePtr frame_curr = std::make_shared<Frame>(cam_, false, nullptr);
	this->saveFrame(frame_curr);
	
	// 이미지 undistort (KITTI라서 할 필요 X)
	cv::Mat img_undist;
	if(system_flags_.flagDoUndistortion) 
	{
		cam_->undistortImage(img, img_undist);
		img_undist.convertTo(img_undist, CV_8UC1);
	}
	else 
		img.copyTo(img_undist);

	// frame_curr에 img_undist, pose, dT01, 시간 부여
	frame_curr->setImageAndTimestamp(img_undist, timestamp);
	frame_curr->setPose(pose);
	frame_curr->setPoseDiff10(dT01.inverse());

	// Get previous and current images
	const cv::Mat& I0 = frame_prev_->getImage();
	const cv::Mat& I1 = frame_curr->getImage();

	std::cout << "OK!\n";

	if( !system_flags_.flagVOInit ) { // 초기화 미완료
		if( !system_flags_.flagFirstImageGot ) { // 최초 이미지	

			// Extract pixels
			PixelVec       pts1;
			LandmarkPtrVec lms1;
			extractor_->resetWeightBin();
			extractor_->extractORBwithBinning(I1, pts1, true);

#ifdef RECORD_EXECUTION_STAT
	statcurr_execution.time_new = 0;
#endif

#ifdef RECORD_LANDMARK_STAT
	// get statistics
	uint32_t n_pts = pts1.size();
	statcurr_landmark.n_initial = n_pts;
	statcurr_landmark.n_pass_bidirection = n_pts;
	statcurr_landmark.n_pass_1p = n_pts;
	statcurr_landmark.n_pass_5p = n_pts;
	statcurr_landmark.n_new = n_pts;
	statcurr_landmark.n_final = n_pts;
	
	// statcurr_landmark.max_age = 1;
	// statcurr_landmark.min_age = 1;
	statcurr_landmark.avg_age = 1.0f;

	statcurr_landmark.n_ok_parallax = 0;
	// statcurr_landmark.min_parallax  = 0.0;
	// statcurr_landmark.max_parallax  = 0.0;
	statcurr_landmark.avg_parallax  = 0.0;
#endif

#ifdef RECORD_EXECUTION_STAT
	statcurr_execution.time_track = statcurr_execution.time_new;
	statcurr_execution.time_1p    = statcurr_execution.time_new;
	statcurr_execution.time_5p    = statcurr_execution.time_new;
#endif
			// 초기 landmark 생성
			lms1.reserve(pts1.size());
			for(auto p : pts1) lms1.push_back(std::make_shared<Landmark>(p, frame_curr, cam_));
			
			// Related Landmark와 tracked pixels를 업데이트
			frame_curr->setPtsSeenAndRelatedLandmarks(pts1,lms1);
			
			this->saveLandmarks(lms1);	

			if( true )
				this->showTracking("img_features", I1, pts1, PixelVec(), PixelVec());

			// 첫 이미지 업데이트 완료
			system_flags_.flagFirstImageGot = true;
		}
		else { 
			// 초기화 미완료
			LandmarkTracking lmtrack_prev;
			lmtrack_prev.pts0 = frame_prev_->getPtsSeen();
			lmtrack_prev.pts1 = PixelVec();
			lmtrack_prev.lms  = frame_prev_->getRelatedLandmarkPtr();

			// 이전 자세의 변화량을 가져온다. 
			PoseSE3 Twc_prev   = frame_prev_->getPose();
			PoseSE3 Tcw_prev   = Twc_prev.inverse();

			PoseSE3 dT10       = frame_curr->getPoseDiff10();
			PoseSE3 dT01_prior = frame_curr->getPoseDiff01();
			PoseSE3 Twc_prior  = frame_curr->getPose();
			PoseSE3 Tcw_prior  = frame_curr->getPose().inverse();

#ifdef RECORD_LANDMARK_STAT
	statcurr_landmark.n_initial = lmtrack_prev.pts0.size();
#endif

#ifdef RECORD_EXECUTION_STAT
	timer::tic();
#endif
			// frame_prev_ 의 lms 를 현재 이미지로 track. 5ms

			// Make tracking prior 
			lmtrack_prev.pts1.resize(lmtrack_prev.pts0.size());
			for(int i = 0; i < lmtrack_prev.pts0.size(); ++i){
				const LandmarkPtr& lm = lmtrack_prev.lms[i];
				if(lm->isTriangulated() && lm->getMaxParallax() > 0.3*D2R){
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
			
#ifdef RECORD_EXECUTION_STAT
	statcurr_execution.time_track = timer::toc(false);
#endif
#ifdef RECORD_LANDMARK_STAT
statcurr_landmark.n_pass_bidirection = lmtrack_kltok.pts0.size();
#endif


#ifdef RECORD_EXECUTION_STAT 
timer::tic(); 
#endif
#ifdef RECORD_EXECUTION_STAT
statcurr_execution.time_1p = timer::toc(false);
#endif
#ifdef RECORD_LANDMARK_STAT
statcurr_landmark.n_pass_1p = lmtrack_scaleok.pts0.size();
#endif


#ifdef RECORD_EXECUTION_STAT
timer::tic();
#endif
			// 5-point algorithm 2ms
			MaskVec mask_5p(lmtrack_scaleok.pts0.size(),true);
			PointVec X0_inlier(lmtrack_scaleok.pts0.size());
			Rot3 dR10;
			Pos3 dt10;
			// if( !motion_estimator_->calcPose5PointsAlgorithm(lmtrack_scaleok.pts0, lmtrack_scaleok.pts1, cam_, dR10, dt10, X0_inlier, mask_5p) ) {
			// 	throw std::runtime_error("calcPose5PointsAlgorithm() is failed.");
			// }

			// Check sampson distance 0.01 ms
			std::vector<float> symm_epi_dist;
			motion_estimator_->calcSampsonDistance(lmtrack_scaleok.pts0, lmtrack_scaleok.pts1, cam_, dT10.block<3,3>(0,0), dT10.block<3,1>(0,3), symm_epi_dist);
			MaskVec mask_sampson(lmtrack_scaleok.pts0.size(),true);
			for(int i = 0; i < mask_sampson.size(); ++i)
				mask_sampson[i] = mask_5p[i] && (symm_epi_dist[i] < THRES_SAMPSON);
			
			LandmarkTracking lmtrack_final;
			this->pruneInvalidLandmarks(lmtrack_scaleok, mask_sampson, lmtrack_final);

			for(int i = 0; i < lmtrack_final.pts0.size(); ++i)
				lmtrack_final.lms[i]->addObservationAndRelatedFrame(lmtrack_final.pts1[i], frame_curr);
				
#ifdef RECORD_EXECUTION_STAT
statcurr_execution.time_5p = timer::toc(false);
#endif				

			// Steering angle을 계산한다.
			float steering_angle_curr = scale_estimator_->calcSteeringAngleFromRotationMat(dT01.block<3,3>(0,0));
			frame_curr->setSteeringAngle(steering_angle_curr);

			// Detect turn region by a steering angle.
			if(scale_estimator_->detectTurnRegions(frame_curr)){
				FramePtrVec frames_turn_tmp;
				frames_turn_tmp = scale_estimator_->getAllTurnRegions();
				for(auto f :frames_turn_tmp)
					stat_.stat_turn.turn_regions.push_back(f);
			}

#ifdef RECORD_FRAME_STAT
statcurr_frame.steering_angle = steering_angle_curr;
#endif

#ifdef RECORD_FRAME_STAT
statcurr_frame.Twc = frame_curr->getPose();
statcurr_frame.Tcw = frame_curr->getPose().inverse();
statcurr_frame.dT_10 = frame_curr->getPoseDiff10();
statcurr_frame.dT_01 = frame_curr->getPoseDiff01();
#endif

#ifdef RECORD_LANDMARK_STAT
statcurr_landmark.n_pass_5p = lmtrack_final.pts0.size();
#endif

			// lmvec1_final 중, depth가 복원되지 않은 경우 복원해준다.
#ifdef RECORD_LANDMARK_STAT
statcurr_landmark.n_ok_parallax = 0;
#endif

#ifdef RECORD_EXECUTION_STAT
timer::tic();
#endif
			// 빈 곳에 특징점 pts1_new 를 추출한다. 2 ms
			PixelVec pts1_new;
			extractor_->updateWeightBin(lmtrack_final.pts1); // 이미 pts1가 있는 곳은 제외.
			extractor_->extractORBwithBinning(I1, pts1_new, true);

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
						LandmarkPtr ptr = std::make_shared<Landmark>(p0_new, frame_prev_, cam_);
						ptr->addObservationAndRelatedFrame(p1_new, frame_curr);

						lmtrack_final.pts0.push_back(p0_new);
						lmtrack_final.pts1.push_back(p1_new);
						lmtrack_final.lms.push_back(ptr);
						this->saveLandmark(ptr);
					}
				}
			}

#ifdef RECORD_EXECUTION_STAT
statcurr_execution.time_new = timer::toc(false);
statcurr_execution.time_total = statcurr_execution.time_new + statcurr_execution.time_track + statcurr_execution.time_1p + statcurr_execution.time_5p;
#endif		
#ifdef RECORD_LANDMARK_STAT
statcurr_landmark.n_new = pts1_new.size();
#endif
			if( true )
				this->showTracking("img_features", I1, lmtrack_final.pts0, lmtrack_final.pts1, pts1_new);
			
			
			// lms1_final 중, depth가 복원되지 않은 경우 복원해준다.
			uint32_t cnt_recon = 0 ;
			for(auto lm : lmtrack_final.lms){
				if( lm->getLastParallax() >= 0.1f*D2R){
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
			frame_curr->setPtsSeenAndRelatedLandmarks(lmtrack_final.pts1,lmtrack_final.lms);

#ifdef RECORD_LANDMARK_STAT
	statcurr_landmark.n_final = lmtrack_final.pts1.size();
#endif

			float avg_age = calcLandmarksMeanAge(lmtrack_final.lms);
#ifdef RECORD_LANDMARK_STAT
	statcurr_landmark.avg_age = avg_age;
#endif

#ifdef RECORD_LANDMARK_STAT
	statcurr_landmark.avg_parallax  = 0;
	statcurr_landmark.n_ok_parallax = 0;
#endif

			// system_flags_.flagVOInit = true;

		}
	}
	else { // VO initialized. Do track the new image.
	}

	// 

	// Update statistics
	stat_.stats_landmark.push_back(statcurr_landmark);
	stat_.stats_frame.push_back(statcurr_frame);
	stat_.stats_execution.push_back(statcurr_execution);
	std::cout << "Statistics Updated. size: " << stat_.stats_landmark.size() << "\n";

	// Check keyframe update rules.
	if(keyframes_->checkUpdateRule(frame_curr)){
		keyframes_->addNewKeyframe(frame_curr);
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
