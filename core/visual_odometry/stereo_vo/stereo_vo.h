#ifndef _STEREO_VO_H_
#define _STEREO_VO_H_

// Set verbose
// #define VERBOSE_STEREO_VO
// #define VERBOSE_STEREO_VO_MORE_SPECIFIC

#include <iostream>
#include <exception>
#include <string>
#include <fstream>

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

#include "core/visual_odometry/ba_solver/sparse_bundle_adjustment.h"

#include "core/util/image_processing.h"
#include "core/util/triangulate_3d.h"

#include "core/util/timer.h"
#include "core/util/cout_color.h"

class StereoVO
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

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
			float thres_alive_ratio = 0.7;
			float thres_mean_parallax = 3.0 * D2R;
			float thres_trans = 1.0;					// meter
			float thres_rotation = 3.0 * D2R; // radian
			int n_max_keyframes_in_window = 9;
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
			PoseSE3 Twc;
			PoseSE3 Tcw;
			PoseSE3 dT_01; // motion from prev to curr
			PoseSE3 dT_10; // motion from curr to prev
			PointVec mappoints;

			FrameStatistics()
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
			PoseSE3 Twc;
			PointVec mappoints;

			KeyframeStatistics()
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

	// Cameras (left, right)
private:
	StereoCameraPtr stereo_cam_;

	// Modules
private:
	std::shared_ptr<FeatureExtractor> extractor_;
	std::shared_ptr<FeatureTracker> tracker_;
	std::shared_ptr<MotionEstimator> motion_estimator_;				// Stereo mode
	std::shared_ptr<SparseBundleAdjustmentSolver> ba_solver_; // Stereo mode

	// For tracker
private:
	StereoFramePtr stframe_prev_;
	cv::Mat previous_left_image_;
	cv::Mat previous_right_image_;

	// For keyframes
private:
	std::shared_ptr<StereoKeyframes> stkeyframes_;

	// All frames and landmarks
private:
	LandmarkPtrVec all_landmarks_;
	StereoFramePtrVec all_stframes_;
	StereoFramePtrVec all_stkeyframes_;

	// debug image.
private:
	cv::Mat img_debug_;

	// Constructor, destructor and track function
public:
	StereoVO(std::string mode, std::string directory_intrinsic);
	~StereoVO() noexcept(false);

	// Tracking function.
public:
	void trackStereoImages(const cv::Mat &img_left, const cv::Mat &img_right, const double &timestamp);

	// Get statistics
public:
	const AlgorithmStatistics &getStatistics() const;

private:
	void showTracking(const std::string &window_name, const cv::Mat &img, const PixelVec &pts0, const PixelVec &pts1, const PixelVec &pts1_new);
	void showTrackingBA(const std::string &window_name, const cv::Mat &img, const PixelVec &pts1, const PixelVec &pts1_project);

public:
	const cv::Mat &getDebugImage();

private:
	void saveLandmark(const LandmarkPtr &lm);
	void saveLandmarks(const LandmarkPtrVec &lms);
	void saveStereoFrame(const StereoFramePtr &stframe);
	void saveStereoFrames(const StereoFramePtrVec &stframes);
	void saveStereoKeyframe(const StereoFramePtr &stframe);

private:
	void loadStereoCameraIntrinsicAndUserParameters(const std::string &dir);
};

#endif