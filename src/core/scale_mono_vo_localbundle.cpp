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

	if( !system_flags_.flagVOInit ) { // 초기화 미완료
		if( !system_flags_.flagFirstImageGot ) { // 최초 이미지	
			// Get the first image
			const cv::Mat& I0 = frame_curr->getImage();

			// Extract pixels
			PixelVec       pts0;
			LandmarkPtrVec lms0;

			extractor_->resetWeightBin();
			extractor_->extractORBwithBinning(I0, pts0);
#ifdef RECORD_EXECUTION_STAT
	statcurr_execution.time_new = 0;
#endif

#ifdef RECORD_LANDMARK_STAT
	// get statistics
	uint32_t n_pts = pts0.size();
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
	statcurr_landmark.avg_parallax  = 0.0;
#endif

#ifdef RECORD_EXECUTION_STAT
	statcurr_execution.time_track = statcurr_execution.time_new;
	statcurr_execution.time_1p    = statcurr_execution.time_new;
	statcurr_execution.time_5p    = statcurr_execution.time_new;
	statcurr_execution.time_localba = statcurr_execution.time_new;
#endif
			// 초기 landmark 생성
			lms0.reserve(pts0.size());
			for(auto p : pts0) 
				lms0.push_back(std::make_shared<Landmark>(p, frame_curr));
			
			// Related Landmark와 tracked pixels를 업데이트
			frame_curr->setPtsSeen(pts0);
			frame_curr->setRelatedLandmarks(lms0);

			frame_curr->setPose(PoseSE3::Identity());
			frame_curr->setPoseDiff10(PoseSE3::Identity());
			
			this->saveLandmarks(lms0);	

			if( true )
				this->showTracking("img_features", frame_curr->getImage(), pts0, PixelVec(), PixelVec());

			// 첫 이미지 업데이트 완료
			system_flags_.flagFirstImageGot = true;
		}
		else {
			// 최초 첫 이미지는 들어왔으나, 아직 초기화가 되지 않은 상태.
			// 이전 프레임의 pixels 와 lms0을 가져온다.
			const PixelVec&       pts0 = frame_prev_->getPtsSeen();
			const LandmarkPtrVec& lms0 = frame_prev_->getRelatedLandmarkPtr();
			const cv::Mat&        I0   = frame_prev_->getImage();
			const cv::Mat&        I1   = frame_curr->getImage();

#ifdef RECORD_LANDMARK_STAT
	statcurr_landmark.n_initial = pts0.size();
#endif

#ifdef RECORD_EXECUTION_STAT
	timer::tic();
#endif
			// frame_prev_ 의 lms 를 현재 이미지로 track.
			PixelVec pts1_track;
			MaskVec  maskvec1_track;
			tracker_->trackBidirection(I0, I1, pts0, params_.feature_tracker.window_size, params_.feature_tracker.max_level, params_.feature_tracker.thres_error, params_.feature_tracker.thres_bidirection,
							           pts1_track, maskvec1_track);
#ifdef RECORD_EXECUTION_STAT
	statcurr_execution.time_track = timer::toc(false);
#endif

			// Tracking 결과를 반영하여 pts1_alive, lms1_alive를 정리한다.
			PixelVec       pts0_alive;
			PixelVec       pts1_alive;
			LandmarkPtrVec lms1_alive;
			int cnt_alive = 0;
			for(int i = 0; i < pts1_track.size(); ++i){
				if( maskvec1_track[i]) {
					pts0_alive.push_back(pts0[i]);
					pts1_alive.push_back(pts1_track[i]);
					lms1_alive.push_back(lms0[i]);
					++cnt_alive;
				}
				else lms0[i]->setDead(); // track failed. Dead point.
			}
		
#ifdef RECORD_LANDMARK_STAT
statcurr_landmark.n_pass_bidirection = cnt_alive;
#endif


#ifdef RECORD_EXECUTION_STAT 
timer::tic(); 
#endif
			// 1-point RANSAC 을 이용하여 outlier를 제거 & tentative steering angle 구함.
			MaskVec maskvec_1p;
			float steering_angle_curr = motion_estimator_->findInliers1PointHistogram(pts0_alive, pts1_alive, cam_, maskvec_1p);
			frame_curr->setSteeringAngle(steering_angle_curr);

			// Detect turn region by a steering angle.
			if(scale_estimator_->detectTurnRegions(frame_curr)){
				FramePtrVec frames_turn_tmp;
				frames_turn_tmp = scale_estimator_->getAllTurnRegions();
				for(auto f :frames_turn_tmp)
					stat_.stat_turn.turn_regions.push_back(f);
			}


#ifdef RECORD_EXECUTION_STAT
statcurr_execution.time_1p = timer::toc(false);
#endif
#ifdef RECORD_FRAME_STAT
statcurr_frame.steering_angle = steering_angle_curr;
#endif
			PixelVec       pts0_1p;
			PixelVec       pts1_1p;
			LandmarkPtrVec lms1_1p;
			int cnt_1p = 0;
			for(int i = 0; i < maskvec_1p.size(); ++i){
				if( maskvec_1p[i]) {
					pts0_1p.push_back(pts0_alive[i]);
					pts1_1p.push_back(pts1_alive[i]);
					lms1_1p.push_back(lms1_alive[i]);
					++cnt_1p;
				}
				else lms1_alive[i]->setDead(); // track failed. Dead point.
			}


#ifdef RECORD_LANDMARK_STAT
statcurr_landmark.n_pass_1p = cnt_1p;
#endif


#ifdef RECORD_EXECUTION_STAT
timer::tic();
#endif
			// pts0 와 pts1을 이용, 5-point algorithm 으로 모션 & X0 를 구한다.
			// 만약 mean optical flow의 중간값이 약 1 px 이하인 경우, 정지 상태로 가정하고 스킵.
			MaskVec maskvec_inlier(pts0_1p.size());
			PointVec X0_inlier(pts0_1p.size());
			Rot3 dR10;
			Pos3 dt10;
			if( !motion_estimator_->calcPose5PointsAlgorithm(pts0_1p, pts1_1p, cam_, dR10, dt10, X0_inlier, maskvec_inlier) ) {
				throw std::runtime_error("calcPose5PointsAlgorithm() is failed.");
			}
#ifdef RECORD_EXECUTION_STAT
statcurr_execution.time_5p = timer::toc(false);
#endif			
			// Frame_curr의 자세를 넣는다.
			float scale;
			if(frame_curr->getID() > 300) scale = 0.22;
			else scale = 0.90;
			PoseSE3 dT10; dT10 << dR10, scale*dt10, 0.0f, 0.0f, 0.0f, 1.0f;
			PoseSE3 dT01 = dT10.inverse();

			frame_curr->setPose(frame_prev_->getPose()*dT01);		
			frame_curr->setPoseDiff10(dT10);		

			// tracking, 5p algorithm, newpoint 모두 합쳐서 살아남은 점만 frame_curr에 넣는다
			float avg_flow = 0.0f;
			PixelVec       pts0_final;
			PixelVec       pts1_final;
			LandmarkPtrVec lms1_final;
			cnt_alive = 0;
			int cnt_parallax_ok = 0;
			for(int i = 0; i < pts0_1p.size(); ++i){
				if( maskvec_inlier[i] ) {
					lms1_1p[i]->addObservationAndRelatedFrame(pts1_1p[i], frame_curr);
					avg_flow += lms1_1p[i]->getAvgOptFlow();
					if(lms1_1p[i]->getMaxParallax() > params_.map_update.thres_parallax) {
						++cnt_parallax_ok;
						// lms1_1p[i]->set3DPoint(X0_inlier[i]);
					}
					pts0_final.push_back(pts0_1p[i]);
					pts1_final.push_back(pts1_1p[i]);
					lms1_final.push_back(lms1_1p[i]);
					++cnt_alive;
				}
				else lms1_1p[i]->setDead(); // 5p algorithm failed. Dead point.
			}
			avg_flow /= (float) cnt_alive;
			std::cout << " AVERAGE FLOW : " << avg_flow << " px\n";
			std::cout << " Parallax OK : " << cnt_parallax_ok << std::endl;

#ifdef RECORD_LANDMARK_STAT
statcurr_landmark.n_pass_5p = cnt_alive;
statcurr_landmark.n_ok_parallax = cnt_parallax_ok;
#endif

			// 깊이를 가진 점 갯수를 세어보고, 30개 이상이면 local bundle을 수행한다.
			uint32_t cnt_depth_ok = 0;
			PixelVec pts1_depth_ok; pts1_depth_ok.reserve(lms1_final.size());
			PointVec X_depth_ok; X_depth_ok.reserve(lms1_final.size());
			for(auto lm : lms1_final){
				if(lm->isTriangulated()){ 
					++cnt_depth_ok;
					pts1_depth_ok.push_back(lm->getObservations().back());
					X_depth_ok.push_back(lm->get3DPoint());
				}
			}
			std::cout << frame_curr->getID() <<" -th image. depth newly reconstructed: " << cnt_depth_ok << std::endl;

			if(cnt_depth_ok >= 20){
				// Do Local BA
				std::cout << " DO Local Bundle Adjustment...\n";

				Rot3 Rwc_est = frame_curr->getPose().block<3,3>(0,0);
				Pos3 twc_est = frame_curr->getPose().block<3,1>(0,3);

				MaskVec maskvec_ba;
				motion_estimator_->calcPoseLocalBundleAdjustment(X_depth_ok, pts1_depth_ok, cam_, Rwc_est, twc_est, maskvec_ba);
				
				PoseSE3 Twc_ba;
				Twc_ba <<Rwc_est, twc_est,0,0,0,1;
				frame_curr->setPose(Twc_ba);
			}

#ifdef RECORD_FRAME_STAT
statcurr_frame.Twc = frame_curr->getPose();
statcurr_frame.Tcw = frame_curr->getPose().inverse();
statcurr_frame.dT_10 = frame_curr->getPoseDiff10();
statcurr_frame.dT_01 = frame_curr->getPoseDiff01();
#endif

			// lms1_final 중, depth가 복원되지 않은 경우 복원해준다.
			PointVec X_world_recon;
			for(auto lm : lms1_final){
				if(!lm->isTriangulated()){
					// std::cout << "parallax: " << lm->getMaxParallax()*R2D << " deg\n";
					if(lm->getMaxParallax() >= 0.5f*D2R){
						// const Pixel& pt0 = *(lm->getObservations().end()-2);
						if(lm->getObservations().size() != lm->getRelatedFramePtr().size())
							throw std::runtime_error("lm->getObservations().size() != lm->getRelatedFramePtr().size()\n");

						uint32_t idx_end = lm->getAge()-1;
						const Pixel& pt0 = lm->getObservations()[idx_end-1];
						// const Pixel& pt0 = lm->getObservations().front();
						const Pixel& pt1 = lm->getObservations().back();
						const PoseSE3& Tw0 = lm->getRelatedFramePtr()[idx_end-1]->getPose();
						// const PoseSE3& Tw0 = lm->getRelatedFramePtr().front()->getPose();
						const PoseSE3& Tw1 = lm->getRelatedFramePtr().back()->getPose();
						PoseSE3 T10 =  Tw1.inverse() * Tw0;

						Point X0, X1;
						Mapping::triangulateDLT(pt0, pt1, T10.block<3,3>(0,0), T10.block<3,1>(0,3), cam_, X0, X1);
						Point Xworld = Tw0.block<3,3>(0,0)*X0 + Tw0.block<3,1>(0,3);
						lm->set3DPoint(Xworld);
						X_world_recon.push_back(Xworld);
					}
				}
			}
			std::cout << " Recon done.\n";

#ifdef RECORD_FRAME_STAT
statcurr_frame.mappoints = X_world_recon;
#endif
#ifdef RECORD_EXECUTION_STAT
timer::tic();
#endif
			// 빈 곳에 특징점 pts1_new 를 추출한다.
			PixelVec pts1_new;
			extractor_->updateWeightBin(pts1_final); // 이미 pts1가 있는 곳은 제외.
			extractor_->extractORBwithBinning(frame_curr->getImage(), pts1_new);
#ifdef RECORD_EXECUTION_STAT
statcurr_execution.time_new = timer::toc(false);
statcurr_execution.time_total = statcurr_execution.time_new + statcurr_execution.time_track + statcurr_execution.time_1p + statcurr_execution.time_5p;
#endif		

#ifdef RECORD_LANDMARK_STAT
statcurr_landmark.n_new = pts1_new.size();
#endif
			if( true )
				this->showTracking("img_features", frame_curr->getImage(), pts0_final, pts1_final, pts1_new);
			
			if( pts1_new.size() > 0 ){
				// 새로운 특징점은 새로운 landmark가 된다.
				for(auto p1_new : pts1_new) {
					LandmarkPtr ptr = std::make_shared<Landmark>(p1_new, frame_curr);
					pts1_final.emplace_back(p1_new);
					lms1_final.push_back(ptr);
					this->saveLandmarks(ptr);	
				}
			}

			// lms1와 pts1을 frame_curr에 넣는다.
			frame_curr->setPtsSeen(pts1_final);
			frame_curr->setRelatedLandmarks(lms1_final);
#ifdef RECORD_LANDMARK_STAT
	statcurr_landmark.n_final = pts1_final.size();
#endif

			float avg_age = calcLandmarksMeanAge(lms1_final);
#ifdef RECORD_LANDMARK_STAT
	statcurr_landmark.avg_age = avg_age;
#endif
			// 초기화를 완료할지 판단
			// lms1_final가 최초 관측되었던 (keyframe) 
			bool initialization_done = false;
			int n_lms_keyframe    = keyframe_->getRelatedLandmarkPtr().size();
			int n_lms_alive       = 0;
			int n_lms_parallax_ok = 0;
			float mean_parallax   = 0;
			for(int i = 0; i < lms1_final.size(); ++i){
				const LandmarkPtr& lm = lms1_final[i];
				if( lm->getRelatedFramePtr().front()->getID() == 0 ) {
					++n_lms_alive;
					mean_parallax += lm->getMaxParallax();
					if(lm->getMaxParallax() >= params_.map_update.thres_parallax){
						++n_lms_parallax_ok;
					}
				}
			}
			mean_parallax /= (float)n_lms_alive;

#ifdef RECORD_LANDMARK_STAT
	statcurr_landmark.avg_parallax = mean_parallax;
	statcurr_landmark.n_ok_parallax = n_lms_parallax_ok;
#endif

			if(mean_parallax > params_.keyframe_update.thres_mean_parallax*110000)
				initialization_done = true;
			
			if(initialization_done){ // lms_tracked_ 의 평균 parallax가 특정 값 이상인 경우, 초기화 끝. 
				// lms_tracked_를 업데이트한다. 
				system_flags_.flagVOInit = true;

				std::cout << "VO initialzed!\n";
			}
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

	// Replace the 'frame_prev_' with 'frame_curr'
	frame_prev_ = frame_curr;

	// Notify a thread.

	mut_scale_estimator_->lock();
	*flag_do_ASR_ = true;
	mut_scale_estimator_->unlock();
	cond_var_scale_estimator_->notify_all();
};