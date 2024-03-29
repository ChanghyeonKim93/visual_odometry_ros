#ifndef _MONO_VO_H_
#define _MONO_VO_H_

#define RECORD_LANDMARK_STAT	// Recording the statistics
#define RECORD_FRAME_STAT			// Recording the statistics
#define RECORD_KEYFRAME_STAT	// Recording the statistics
#define RECORD_EXECUTION_STAT // Recording the statistics

#include <iostream>
#include <exception>
#include <string>
#include <fstream>

#include <thread>
#include <mutex>
#include <condition_variable>

// Eigen
#include "eigen3/Eigen/Dense"

// OpenCV
#include "opencv4/opencv2/core.hpp"
#include "opencv4/opencv2/imgproc.hpp"
#include "opencv4/opencv2/highgui.hpp"
#include "opencv4/opencv2/core/eigen.hpp"

#define CV_GRAY2RGB cv::COLOR_GRAY2BGR

// Defines
#include "core/defines/define_macro.h"
#include "core/defines/define_type.h"

// custom
#include "core/visual_odometry/camera.h"

#include "core/visual_odometry/frame.h"
#include "core/visual_odometry/landmark.h"
#include "core/visual_odometry/keyframes.h"

#include "core/visual_odometry/feature_extractor.h"
#include "core/visual_odometry/feature_tracker.h"
#include "core/visual_odometry/motion_estimator.h"

#include "core/util/image_processing.h"
#include "core/util/triangulate_3d.h"

#include "core/util/timer.h"
#include "core/util/cout_color.h"

class MonoVO
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	// Camera object
private:
	std::shared_ptr<Camera> cam_;

	// Modules
private:
	std::shared_ptr<FeatureExtractor> extractor_;
	std::shared_ptr<FeatureTracker> tracker_;
	std::shared_ptr<MotionEstimator> motion_estimator_;

private:
	struct SystemFlags
	{
		bool flagFirstImageGot;
		bool flagVOInit;
		bool flagDoUndistortion;
		SystemFlags() : flagFirstImageGot(false), flagVOInit(false), flagDoUndistortion(false){};
	};

	struct AlgorithmParameters
	{
		struct FeatureTrackerParameters
		{
			float thres_error = 125.0;		 // KLT error threshold
			float thres_bidirection = 1.0; // bidirection pixel error threshold
			float thres_sampson = 10.0;
			int window_size = 15; // KLT window size
			int max_level = 6;		// KLT maximum pyramid level
		};
		struct FeatureExtractorParameters
		{
			int n_features = 100;					// # of features to extract from a bucket
			int n_bins_u = 16;						// Bucket grid size u
			int n_bins_v = 8;							// Bucket grid size v
			float thres_fastscore = 25.0; // FAST score threshold
			float radius = 15.0;					// NONMAX pixel threshold
		};
		struct MotionEstimatorParameters
		{
			float thres_1p_error = 10.0;		// sampson error threshold
			float thres_5p_error = 2.0;			// sampson error threshold
			float thres_poseba_error = 5.0; // reprojection error.
		};
		struct KeyframeUpdateParameters
		{
			float thres_translation = 1.0;		// meter
			float thres_rotation = 3.0 * D2R; // radian
			float thres_overlap_ratio = 0.7;
			int n_max_keyframes_in_window = 7;
		};
		struct MappingParameters
		{
			float thres_parallax = 1.0 * D2R;
		};

		FeatureTrackerParameters feature_tracker;
		FeatureExtractorParameters feature_extractor;
		MotionEstimatorParameters motion_estimator;
		KeyframeUpdateParameters keyframe_update;
		MappingParameters map_update;
	};

public:
	struct AlgorithmStatistics
	{
		struct LandmarkStatistics
		{
			int n_initial;					// the number of landmarks tracked on the current frame from the previous frame.
			int n_pass_bidirection; // bidirectional KLT tracking
			int n_pass_1p;
			int n_pass_5p;
			int n_new;
			int n_final; // remained number of landmarks on the current frame.

			int max_age;	 // max. age of landmark observed in current frame
			int min_age;	 // min. age of landmark observed in current frame
			float avg_age; // avg. age of landmark observed in current frame

			int n_ok_parallax;
			float min_parallax;
			float max_parallax;
			float avg_parallax;

			LandmarkStatistics() : n_initial(0), n_pass_bidirection(0), n_pass_1p(0), n_pass_5p(0), n_new(0), n_final(0),
														 max_age(0), min_age(0), avg_age(0.0f),
														 n_ok_parallax(0), min_parallax(0.0f), max_parallax(0.0f), avg_parallax(0.0f){};
		};

		struct FrameStatistics
		{
			float steering_angle; // steering angle from prev to curr
			PoseSE3 Twc;
			PoseSE3 Tcw;
			PoseSE3 dT_01; // motion from prev to curr
			PoseSE3 dT_10; // motion from curr to prev
			PointVec mappoints;

			FrameStatistics() : steering_angle(0.0f)
			{
				Twc = PoseSE3::Identity();
				Tcw = PoseSE3::Identity();
				dT_01 = PoseSE3::Identity();
				dT_10 = PoseSE3::Identity();
				mappoints.reserve(500);
			};
		};

		struct KeyframeStatistics
		{
			float steering_angle; // steering angle from prev to curr
			PoseSE3 Twc;
			PointVec mappoints;

			KeyframeStatistics() : steering_angle(0.0f)
			{
				Twc = PoseSE3::Identity();
				mappoints.reserve(500);
			};
		};

		struct ExecutionStatistics
		{
			float time_track;		// execution time per frame
			float time_1p;			// execution time per frame
			float time_5p;			// execution time per frame
			float time_localba; // execution time per frame
			float time_new;			// execution time per frame
			float time_total;		// execution time per frame

			ExecutionStatistics() : time_total(0.0f),
															time_track(0.0f), time_1p(0.0f), time_5p(0.0f), time_localba(0.0f), time_new(0.0f){};
		};

		std::vector<LandmarkStatistics> stats_landmark;
		std::vector<FrameStatistics> stats_frame;
		std::vector<KeyframeStatistics> stats_keyframe;
		std::vector<ExecutionStatistics> stats_execution;

		AlgorithmStatistics()
		{
			stats_landmark.reserve(5000);
			stats_frame.reserve(5000);
			stats_keyframe.reserve(5000);
			stats_execution.reserve(5000);
		};
	};

	// Parameters
private:
	AlgorithmParameters params_;

	// Statistics
private:
	AlgorithmStatistics stat_;

	// For algorithm state
private:
	SystemFlags system_flags_;

	// For tracker
private:
	FramePtr frame_prev_;
	FramePtr keyframe_ref_; // reference keyframe for pose-only BA
	cv::Mat previous_image_;

	// For keyframes
private:
	std::shared_ptr<Keyframes> keyframes_;

	// All frames and landmarks
private:
	LandmarkPtrVec all_landmarks_;
	FramePtrVec all_frames_;
	FramePtrVec all_keyframes_;

	// debug image.
private:
	cv::Mat img_debug_;

public:
	MonoVO(std::string mode, std::string directory_intrinsic);
	~MonoVO() noexcept(false);

	void trackImage(const cv::Mat &img, const double &timestamp);
	void trackImageBackend(const cv::Mat &img, const double &timestamp, const PoseSE3 &pose, const PoseSE3 &dT01);

	void trackImageFeatureOnly(const cv::Mat &img, const double &timestamp);

	const AlgorithmStatistics &getStatistics() const;

private:
	int pruneInvalidLandmarks(const PixelVec &pts0, const PixelVec &pts1, const LandmarkPtrVec &lms, const MaskVec &mask,
														PixelVec &pts0_alive, PixelVec &pts1_alive, LandmarkPtrVec &lms_alive);
	int pruneInvalidLandmarks(const LandmarkTracking &lmtrack, const MaskVec &mask,
														LandmarkTracking &lmtrack_alive);
	void updateKeyframe(const FramePtr &frame);
	void saveLandmark(const LandmarkPtr &lm);
	void saveLandmarks(const LandmarkPtrVec &lms);
	void saveFrame(const FramePtr &frame);
	void saveFrames(const FramePtrVec &frames);

	void saveKeyframe(const FramePtr &frame);

private:
	float calcLandmarksMeanAge(const LandmarkPtrVec &lms);

private:
	void showTracking(const std::string &window_name, const cv::Mat &img, const PixelVec &pts0, const PixelVec &pts1, const PixelVec &pts1_new);
	void showTrackingBA(const std::string &window_name, const cv::Mat &img, const PixelVec &pts1, const PixelVec &pts1_project);
	void showTracking(const std::string &window_name, const cv::Mat &img, const LandmarkPtrVec &lms);

public:
	const cv::Mat &getDebugImage();

private:
	void loadCameraIntrinsicAndUserParameters(const std::string &dir);
};

#endif