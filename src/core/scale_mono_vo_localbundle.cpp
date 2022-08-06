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

			// 초기 landmark 생성
			lms0.reserve(pts0.size());
			for(auto p : pts0) 
				lms0.push_back(std::make_shared<Landmark>(p, frame_curr));
			
			// Related Landmark와 tracked pixels를 업데이트
			frame_curr->setPtsSeen(pts0);
			frame_curr->setRelatedLandmarks(lms0);

			frame_curr->setPose(PoseSE3::Identity());
			PoseSE3 T_init = PoseSE3::Identity();
			T_init.block<3,1>(0,3) << 0,0,-0.8;
			frame_curr->setPoseDiff10(T_init);
			
			this->saveLandmarks(lms0);	

			if( true )
				this->showTracking("img_features", frame_curr->getImage(), pts0, PixelVec(), PixelVec());

			// 첫 이미지 업데이트 완료
			system_flags_.flagFirstImageGot = true;
		}
		else {
// INIT
//
//
//
//
//
//
//
//
//
//
			// 초기화 미완료
			// 최초 첫 이미지는 들어왔으나, 아직 초기화가 되지 않은 상태.
			// 이전 프레임의 pixels 와 lms0을 가져온다.
			const PixelVec&       pts0 = frame_prev_->getPtsSeen();
			const LandmarkPtrVec& lms0 = frame_prev_->getRelatedLandmarkPtr();
			const cv::Mat&        I0   = frame_prev_->getImage();
			const cv::Mat&        I1   = frame_curr->getImage();

			// 이전 자세의 변화량을 가져온다. 
			PoseSE3 Twc_prev   = frame_prev_->getPose();
			PoseSE3 Tcw_prev   = Twc_prev.inverse();

			PoseSE3 dT01_prior = frame_prev_->getPoseDiff01();
			PoseSE3 Twc_prior  = Twc_prev*dT01_prior;
			PoseSE3 Tcw_prior  = Twc_prior.inverse();

			// 깊이가 있는 점들에 대해 prior를 생성한다.
			PixelVec pts1_track; pts1_track.resize(lms0.size());
			for(int i = 0; i < lms0.size(); ++i){
				const LandmarkPtr& lm = lms0[i];
				if(lm->isTriangulated() && lm->getCovarianceInverseDepth() < 0.0) {
					Point Xc = Tcw_prior.block<3,3>(0,0)*lm->get3DPoint() + Tcw_prior.block<3,1>(0,3);
					pts1_track[i] = cam_->projectToPixel(Xc);
				}
				else pts1_track[i] = pts0[i];
			}

			// frame_prev_ 의 lms 를 현재 이미지로 track.
			MaskVec  maskvec1_track;
			tracker_->trackBidirectionWithPrior(I0, I1, pts0, params_.feature_tracker.window_size, params_.feature_tracker.max_level, params_.feature_tracker.thres_error, params_.feature_tracker.thres_bidirection,
							           pts1_track, maskvec1_track);

			// Tracking 결과를 반영하여 pts1_alive, lms1_alive를 정리한다.
			PixelVec       pts0_alive0;
			PixelVec       pts1_alive0;
			LandmarkPtrVec lms1_alive0;
			int cnt_alive = 0;
			for(int i = 0; i < pts1_track.size(); ++i){
				if( maskvec1_track[i]) {
					pts0_alive0.push_back(pts0[i]);
					pts1_alive0.push_back(pts1_track[i]);
					lms1_alive0.push_back(lms0[i]);
					++cnt_alive;
				}
				else lms0[i]->setDead(); // track failed. Dead point.
			}

			MaskVec mask_refine(pts0_alive0.size(), true);
			// tracker_->refineScale(I0, I1, frame_curr->getImageDu(), frame_curr->getImageDv(), pts0_alive0, 1.25f, pts1_alive0, mask_refine);
			
			PixelVec       pts0_alive;
			PixelVec       pts1_alive;
			LandmarkPtrVec lms1_alive;
			cnt_alive = 0;
			for(int i = 0; i < pts0_alive0.size(); ++i){
				if( mask_refine[i]) {
					pts0_alive.push_back(pts0_alive0[i]);
					pts1_alive.push_back(pts1_alive0[i]);
					lms1_alive.push_back(lms1_alive0[i]);
					++cnt_alive;
				}
				else lms1_alive0[i]->setDead(); // track failed. Dead point.
			}

			// 깊이를 가진 점 갯수를 세어보고, 30개 이상이면 local bundle을 수행한다.
			uint32_t cnt_depth_ok = 0;
			PixelVec pts1_depth_ok; 
			PointVec X_depth_ok; 
			pts1_depth_ok.reserve(lms1_alive.size());
			X_depth_ok.reserve(lms1_alive.size());

			Rot3 Rcw_prev = Tcw_prev.block<3,3>(0,0);
			Pos3 tcw_prev = Tcw_prev.block<3,1>(0,3);
			LandmarkPtrVec lms1_depthok;
			for(auto lm : lms1_alive){
				if(lm->isTriangulated() && lm->getAge() > 2 && lm->getMaxParallax() > 0.3*D2R){ 
					Point Xc = Rcw_prev*lm->get3DPoint() + tcw_prev;
					if(Xc(2) > 0){
						pts1_depth_ok.push_back(lm->getObservations().back());
						X_depth_ok.push_back(Xc);
						++cnt_depth_ok;
					}
				}
			}
			std::cout << frame_curr->getID() <<" -th image. depth newly reconstructed: " << cnt_depth_ok << std::endl;

			if(cnt_depth_ok > 111000){
				// Do Local BA
				std::cout << " DO Local Bundle Adjustment...\n";

				MaskVec maskvec_ba;
				Rot3 dR01 = dT01_prior.block<3,3>(0,0);
				Pos3 dt01 = dT01_prior.block<3,1>(0,3);

				motion_estimator_->calcPoseOnlyBundleAdjustment(X_depth_ok, pts1_depth_ok, cam_, dR01, dt01, maskvec_ba);
				
				PoseSE3 dT01_ba, dT10_ba;
				dT01_ba << dR01, dt01, 0,0,0,1;
				dT10_ba = dT01_ba.inverse();
				if(!std::isnan(dt01.norm())){
					frame_curr->setPose(Twc_prev*dT01_ba);
					frame_curr->setPoseDiff10(dT10_ba);
				}
			}
			else { // do 8 point algorihtm (scale is of the previous frame)
				// pts0 와 pts1을 이용, 5-point algorithm 으로 모션 & X0 를 구한다.
				// 만약 mean optical flow의 중간값이 약 1 px 이하인 경우, 정지 상태로 가정하고 스킵.
				MaskVec maskvec_inlier(pts0_alive.size());
				PointVec X0_inlier(pts0_alive.size());
				Rot3 dR10;
				Pos3 dt10;
				if( !motion_estimator_->calcPose5PointsAlgorithm(pts0_alive, pts1_alive, cam_, dR10, dt10, X0_inlier, maskvec_inlier) ) {
					throw std::runtime_error("calcPose5PointsAlgorithm() is failed.");
				}

				// Frame_curr의 자세를 넣는다.
				float scale = frame_prev_->getPoseDiff01().block<3,1>(0,3).norm();
				PoseSE3 dT10; dT10 << dR10, (scale/dt10.norm())*dt10, 0.0f, 0.0f, 0.0f, 1.0f;
				PoseSE3 dT01 = dT10.inverse();

				frame_curr->setPose(Twc_prev*dT01);		
				frame_curr->setPoseDiff10(dT10);		
			}

			PixelVec       pts0_final;
			PixelVec       pts1_final;
			LandmarkPtrVec lms1_final;
			int cnt_parallax_ok = 0;
			for(int i = 0; i < pts0_alive.size(); ++i){
				lms1_alive[i]->addObservationAndRelatedFrame(pts1_alive[i], frame_curr);
				
				pts0_final.push_back(pts0_alive[i]);
				pts1_final.push_back(pts1_alive[i]);
				lms1_final.push_back(lms1_alive[i]);
			}

#ifdef RECORD_FRAME_STAT
statcurr_frame.Twc = frame_curr->getPose();
statcurr_frame.Tcw = frame_curr->getPose().inverse();
statcurr_frame.dT_10 = frame_curr->getPoseDiff10();
statcurr_frame.dT_01 = frame_curr->getPoseDiff01();
#endif

			// 빈 곳에 특징점 pts1_new 를 추출한다.
			PixelVec pts1_new;
			extractor_->updateWeightBin(pts1_final); // 이미 pts1가 있는 곳은 제외.
			extractor_->extractORBwithBinning(frame_curr->getImage(), pts1_new);

			if( true )
				this->showTracking("img_features", frame_curr->getImage(), pts0_final, pts1_final, pts1_new);
			
			if( pts1_new.size() > 0 ){

				// 새로운 특징점을 back-track한다.
				PixelVec pts0_new;
				MaskVec mask0_new;
				tracker_->trackBidirection(I1,I0,pts1_new, 15, 5, 20.0, 10.0,
					pts0_new, mask0_new);

				// 새로운 특징점은 새로운 landmark가 된다.
				for(int j = 0; j < pts1_new.size(); ++j) {
					if(mask0_new[j]){
						const Pixel& p0_new = pts0_new[j];
						const Pixel& p1_new = pts1_new[j];
						LandmarkPtr ptr = std::make_shared<Landmark>(p0_new, frame_prev_);

						pts1_final.emplace_back(p1_new);
						lms1_final.push_back(ptr);
						this->saveLandmarks(ptr);	
					}
				}
			}

			// lms1_final 중, depth가 복원되지 않은 경우 복원해준다.
			PointVec X_world_recon;
			uint32_t cnt_recon = 0 ;
			for(auto lm : lms1_final){
				// if(lm->getMaxParallax() < 9.5*D2R){
				// std::cout << "parallax: " << lm->getMaxParallax()*R2D << " deg\n";
				if(lm->getMaxParallax() >= 0.1f*D2R){
					// const Pixel& pt0 = *(lm->getObservations().end()-2);
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

					// Calculate cov
					float parallax_now = lm->getLastParallax();
					float cov_invd_curr = 25.0;
					if(parallax_now <= 3.0*D2R) cov_invd_curr /= (parallax_now*R2D)*(parallax_now*R2D);
					
					Point Xworld = Tw0.block<3,3>(0,0)*X0 + Tw0.block<3,1>(0,3);
					if(X0(2) > 0){
						if(0 && lm->isTriangulated()){
							// Update inverse depth
							lm->updateInverseDepth(1.0f/X0(2), cov_invd_curr);
							Xworld = lm->get3DPoint();
						}
						else{
							lm->setInverseDepth(1.0f/X0(2));
							// std::cout << "invd : "<< lm->getInverseDepth() <<std::endl;
							lm->setCovarianceInverseDepth(cov_invd_curr);					
							lm->set3DPoint(Xworld);
						}
					}
					if(lm->getMaxParallax() > 2.0*D2R)
						X_world_recon.push_back(Xworld);

					++cnt_recon;
				}
			}
			std::cout << " Recon done. : " << cnt_recon << "\n";
#ifdef RECORD_FRAME_STAT
statcurr_frame.mappoints = X_world_recon;
#endif

			// lms1와 pts1을 frame_curr에 넣는다.
			frame_curr->setPtsSeen(pts1_final);
			frame_curr->setRelatedLandmarks(lms1_final);

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