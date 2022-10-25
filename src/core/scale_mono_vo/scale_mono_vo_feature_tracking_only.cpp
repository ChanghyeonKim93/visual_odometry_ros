#include "core/scale_mono_vo/scale_mono_vo.h"

void ScaleMonoVO::trackImageFeatureOnly(const cv::Mat& img, const double& timestamp)
{

	const std::string text_black   = "\033[0;30m";
	const std::string text_red     = "\033[0;31m";
	const std::string text_green   = "\033[0;32m";
	const std::string text_yellow  = "\033[0;33m";
	const std::string text_blue    = "\033[0;34m";
	const std::string text_magenta = "\033[0;35m";
	const std::string text_cyan    = "\033[0;36m";
	const std::string text_white   = "\033[0;37m";

	const std::string cout_reset     = "\033[0m";
	const std::string cout_bold      = "\033[1m";
	const std::string cout_underline = "\033[4m";
	const std::string cout_inverse   = "\033[7m";

	const std::string cout_boldoff      = "\033[21m";
	const std::string cout_underlineoff = "\033[24m";
	const std::string cout_inverseoff   = "\033[27m";


	float THRES_SAMPSON  = params_.feature_tracker.thres_sampson;
	float THRES_PARALLAX = params_.map_update.thres_parallax;on;
			
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
			extractor_->extractORBwithBinning_fast(I1, lmtrack_curr.pts1, true);

			// 초기 landmark 생성
			for(auto pt : lmtrack_curr.pts1)
			{
				LandmarkPtr lm_new = std::make_shared<Landmark>(pt, frame_curr, cam_);
				lmtrack_curr.lms.push_back(lm_new);
			}
			
			// Related Landmark와 tracked pixels를 업데이트
			frame_curr->setPtsSeenAndRelatedLandmarks(lmtrack_curr.pts1, lmtrack_curr.lms);

			frame_curr->setPose(PoseSE3::Identity());
			PoseSE3 T_init = PoseSE3::Identity();
			T_init.block<3,1>(0,3) << 0,0,-1; // get initial scale.
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

			MaskVec mask_scaleok(lmtrack_scaleok.pts0.size(), true);			
			LandmarkTracking lmtrack_final;
			this->pruneInvalidLandmarks(lmtrack_scaleok, mask_scaleok, lmtrack_final);

			// Update tracking results
			for(int i = 0; i < lmtrack_final.lms.size(); ++i)
				lmtrack_final.lms[i]->addObservationAndRelatedFrame(lmtrack_final.pts1[i], frame_curr);
			
			// 빈 곳에 특징점 pts1_new 를 추출한다. 2 ms
			PixelVec pts1_new;
			extractor_->updateWeightBin(lmtrack_final.pts1); // 이미 pts1가 있는 곳은 제외.
			extractor_->extractORBwithBinning_fast(I1, pts1_new, true);

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
						LandmarkPtr ptr = std::make_shared<Landmark>(p0_new, frame_prev_, cam_);
						ptr->addObservationAndRelatedFrame(p1_new, frame_curr);

						lmtrack_final.pts0.push_back(p0_new);
						lmtrack_final.pts1.push_back(p1_new);
						lmtrack_final.lms.push_back(ptr);
						this->saveLandmark(ptr);
					}
				}
			}

			// lms1와 pts1을 frame_curr에 넣는다.
			frame_curr->setPtsSeenAndRelatedLandmarks(lmtrack_final.pts1, lmtrack_final.lms);

			system_flags_.flagVOInit = true;
		}
	}
	else 
	{
		timer::tic();
		// VO initialized. Do track the new image.
		LandmarkTracking lmtrack_prev;
		lmtrack_prev.pts0 = frame_prev_->getPtsSeen();
		lmtrack_prev.pts1 = PixelVec();
		lmtrack_prev.lms  = frame_prev_->getRelatedLandmarkPtr();

		// frame_prev_ 의 lms 를 현재 이미지로 track. 5ms
		timer::tic();
		MaskVec mask_track;
		tracker_->trackBidirection(I0, I1, lmtrack_prev.pts0, params_.feature_tracker.window_size, params_.feature_tracker.max_level, params_.feature_tracker.thres_error, params_.feature_tracker.thres_bidirection,
			lmtrack_prev.pts1, mask_track);
		std::cout << text_green << "Time [track bidirection]: " << timer::toc(0) << " [ms]\n" << cout_reset;

		LandmarkTracking lmtrack_kltok;
		std::cout << "# of kltok : " << this->pruneInvalidLandmarks(lmtrack_prev, mask_track, lmtrack_kltok) << std::endl;

		MaskVec mask_kltok;
		LandmarkTracking lmtrack_final;
		std::cout << "# of samps : " << this->pruneInvalidLandmarks(lmtrack_kltok, mask_sampson, lmtrack_final) << std::endl;

		for(int i = 0; i < lmtrack_final.pts0.size(); ++i)
			lmtrack_final.lms[i]->addObservationAndRelatedFrame(lmtrack_final.pts1[i], frame_curr);
				
#ifdef RECORD_FRAME_STAT
statcurr_frame.Twc   = frame_curr->getPose();
statcurr_frame.Tcw   = frame_curr->getPoseInv();
statcurr_frame.dT_10 = frame_curr->getPoseDiff10();
statcurr_frame.dT_01 = frame_curr->getPoseDiff01();
#endif
			
		// 빈 곳에 특징점 pts1_new 를 추출한다. 2 ms
		timer::tic();
		PixelVec pts1_new;
		extractor_->updateWeightBin(lmtrack_final.pts1); // 이미 pts1가 있는 곳은 제외.
		extractor_->extractORBwithBinning_fast(I1, pts1_new, true);
		std::cout << text_green << "Time [extract ORB      ]: " << timer::toc(0) << " [ms]\n" << cout_reset;
		
		// std::cout << "# features : " << pts1_new.size()  << std::endl;

		timer::tic();
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
					LandmarkPtr ptr = std::make_shared<Landmark>(p0_new, frame_prev_,cam_);
					ptr->addObservationAndRelatedFrame(p1_new, frame_curr);

					lmtrack_final.pts0.push_back(p0_new);
					lmtrack_final.pts1.push_back(p1_new);
					lmtrack_final.lms.push_back(ptr);
					this->saveLandmark(ptr);
				}
			}
		}
		std::cout << text_green << "Time [New fts ext track]: " << timer::toc(0) << " [ms]\n" << cout_reset;
		
		// lms1와 pts1을 frame_curr에 넣는다.
		frame_curr->setPtsSeenAndRelatedLandmarks(lmtrack_final.pts1, lmtrack_final.lms);
	}

	// Check keyframe update rules.
	bool flag_add_new_keyframe = keyframes_->checkUpdateRule(frame_curr);
	if( flag_add_new_keyframe )
	{
		
		timer::tic();
		// Add new keyframe
		this->saveKeyframe(frame_curr);
		keyframes_->addNewKeyframe(frame_curr);

		// Add this frame to the scale estimator
		scale_estimator_->insertNewFrame(frame_curr);

		
		// Make variables for refine tracking
		const LandmarkPtrVec& lms_final = frame_curr->getRelatedLandmarkPtr();
		FloatVec scale_estimated(lms_final.size(), 1.0f);
		MaskVec mask_final(lms_final.size(), true);
		PixelVec pts_refine(lms_final.size());

		for(int i = 0; i < lms_final.size(); ++i){
			// Estimate current image-patch-scale...
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
		std::cout << text_green << "Time [keyframe addition]: " << timer::toc(0) << " [ms]\n" << cout_reset;
		

		// Refine the tracking results
		timer::tic();
		tracker_->refineTrackWithScale(I1, lms_final, scale_estimated, pts_refine, mask_final);
		std::cout << text_green << "Time [trackWithScale   ]: " << timer::toc(0) << " [ms]\n" << cout_reset;

		// Update points
		for(int i = 0; i < lms_final.size(); ++i)
		{
			const LandmarkPtr& lm = lms_final[i];
			if(mask_final[i])
				lm->changeLastObservation(pts_refine[i]);
		}

		// Reconstruct map points
		// lms1_final 중, depth가 복원되지 않은 경우 복원해준다.
		timer::tic();
		uint32_t cnt_recon = 0;
		for(auto lm : frame_curr->getRelatedLandmarkPtr())
		{
			if( !lm->isTriangulated() && lm->getLastParallax() >= THRES_PARALLAX )
			{
				if( lm->getObservationsOnKeyframes().size() > 1 )
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
					if(X0(2) > 0 && X1(2) > 0) {
						Point Xworld = Tw0.block<3,3>(0,0)*X0 + Tw0.block<3,1>(0,3);
						lm->set3DPoint(Xworld);
						++cnt_recon;
					}
				}
			}
		}

		// Do local bundle adjustment for keyframes.
		motion_estimator_->localBundleAdjustmentSparseSolver(keyframes_, cam_);
		std::cout << text_green << "Time [LBA              ]: " << timer::toc(0) << " [ms]\n" << cout_reset;
		






#ifdef RECORD_KEYFRAME_STAT
timer::tic();
PointVec X_tmp;
const LandmarkPtrVec& lmvec_tmp = frame_curr->getRelatedLandmarkPtr();
for(int i = 0; i < lmvec_tmp.size(); ++i)
{
	X_tmp.push_back(lmvec_tmp[i]->get3DPoint());
}
statcurr_keyframe.Twc = frame_curr->getPose();
statcurr_keyframe.mappoints = X_tmp;
stat_.stats_keyframe.push_back(statcurr_keyframe);

for(int j = 0; j < stat_.stats_keyframe.size(); ++j){
	stat_.stats_keyframe[j].Twc = all_keyframes_[j]->getPose();

	const LandmarkPtrVec& lmvec_tmp = all_keyframes_[j]->getRelatedLandmarkPtr();
	stat_.stats_keyframe[j].mappoints.resize(lmvec_tmp.size());
	for(int i = 0; i < lmvec_tmp.size(); ++i) {
		stat_.stats_keyframe[j].mappoints[i] = lmvec_tmp[i]->get3DPoint();
	}
}
std::cout << text_green << "Time [RECORD KEYFR STAT]: " << timer::toc(0) << " [ms]\n" << cout_reset;
#endif





	} // KEYFRAME addition done.



	
	// Replace the 'frame_prev_' with 'frame_curr'
	this->frame_prev_ = frame_curr;

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
	// stat_.stats_frame.resize(0);
	stat_.stats_frame.push_back(statcurr_frame);
	
	for(int j = 0; j < this->all_frames_.size(); ++j){
		stat_.stats_frame[j].Twc = all_frames_[j]->getPose();
	}
	stat_.stats_execution.push_back(statcurr_execution);
	std::cout << "Statistics Updated. size: " << stat_.stats_landmark.size() << "\n";

	// Notify a thread.
	mut_scale_estimator_->lock();
	*flag_do_ASR_ = true;
	mut_scale_estimator_->unlock();
	cond_var_scale_estimator_->notify_all();
};