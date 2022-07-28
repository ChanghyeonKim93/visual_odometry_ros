#ifndef _FEATURE_EXTRACTOR_H_
#define _FEATURE_EXTRACTOR_H_

#include <iostream>
#include <exception>
#include <vector>
#include <string>

// ROS eigen
#include <Eigen/Dense>

// ROS cv_bridge
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

#include "core/type_defines.h"
#include "core/image_processing.h"

class FeatureExtractor ;

struct ParamsORB {
	int   MaxFeatures     ;// % MaxFeatures (300) % The maximum number of features to retain.
	float ScaleFactor     ;// % ScaleFactor (1.2) % Pyramid decimation ratio.
	int   NLevels         ;// % NLevels (8)% The number of pyramid levels.
	float EdgeThreshold   ;// % EdgeThreshold (31)% This is size of the border where the features are not detected.
	int   FirstLevel      ;// % FirstLevel (0)% The level of pyramid to put source image to.
	int   WTA_K           ;// % WTA_K (2)% The number of points that produce each element of the oriented BRIEF descriptor. (2, 3, 4)
	std::string ScoreType ;// % ScoreType (Harris) % Algorithm used to rank features. (Harris, FAST)
	int   PatchSize       ;// % PatchSize (31) % Size of the patch used by the oriented BRIEF descriptor.
	float FastThreshold   ;// % FastThreshold (20)% Threshold on difference between intensity of the central pixel and pixels of a circle around this pixel
	float r               ;// % Radius of non maximum suppresion (5)
	int   n_bins_u        ;
	int   n_bins_v        ;
	ParamsORB(){
		MaxFeatures     = 100;    // % MaxFeatures (300) % The maximum number of features to retain.
		ScaleFactor     = 1.05;   // % ScaleFactor (1.2) % Pyramid decimation ratio.
		NLevels         = 1;      // % NLevels (8)% The number of pyramid levels.
		EdgeThreshold   = 6;      // % EdgeThreshold (31)% This is size of the border where the features are not detected.
		FirstLevel      = 0;      // % FirstLevel (0)% The level of pyramid to put source image to.
		WTA_K           = 2;      // % WTA_K (2)% The number of points that produce each element of the oriented BRIEF descriptor. (2, 3, 4)
	    ScoreType       = "FAST"; // % ScoreType (Harris) % Algorithm used to rank features. (Harris, FAST)
		PatchSize       = 31;     // % PatchSize (31) % Size of the patch used by the oriented BRIEF descriptor.
		FastThreshold   = 15;     // % FastThreshold (20)% Threshold on difference between intensity of the central pixel and pixels of a circle around this pixel
		r               = 5;      // % Radius of non maximum suppresion (5)
		n_bins_u        = -1;
		n_bins_v        = -1;
	};
};

struct WeightBin {
	int* weight ;
	int* u_bound;
	int* v_bound;
	int n_bins_u;
	int n_bins_v;
	int u_step  ;
	int v_step  ;
	int n_bins_total;

	WeightBin() {
		weight  = nullptr;
		u_bound = nullptr;
		v_bound = nullptr;
		n_bins_u = 0;
		n_bins_v = 0;
		u_step = 0;
		v_step = 0;
		n_bins_total = 0;
	};
	~WeightBin() {
		if (weight != nullptr) delete weight;
		if (v_bound != nullptr) delete v_bound;
		if (u_bound != nullptr) delete u_bound;
	};

	void init(int n_cols, int n_rows, int n_bins_u, int n_bins_v) {
		this->n_bins_u = n_bins_u;
		this->n_bins_v = n_bins_v;
		n_bins_total = n_bins_u*n_bins_v;

		weight = new int[this->n_bins_u*this->n_bins_v];
		for (int i = 0; i < this->n_bins_u*this->n_bins_v; ++i) weight[i] = 1;

		this->u_bound = new int[this->n_bins_u + 1];
		this->v_bound = new int[this->n_bins_v + 1];

		this->u_bound[0] = 1;
		this->v_bound[0] = 1;

		this->u_bound[n_bins_u] = n_cols;
		this->v_bound[n_bins_v] = n_rows;

		this->u_step = (int)floor((float)n_cols / (float)n_bins_u);
		this->v_step = (int)floor((float)n_rows / (float)n_bins_v);

		for (int i = 1; i < n_bins_u; ++i) this->u_bound[i] = u_step * i;
		for (int i = 1; i < n_bins_v; ++i) this->v_bound[i] = v_step * i;

		std::cout<< "   - FEATURE_EXTRACTOR - WeightBin - 'init'\n";
	};

	void reset() {
		for (int i = 0; i < n_bins_total; ++i) weight[i] = 1;
	};

	void update(const PixelVec& pts) {
		int n_pts = pts.size();

		for (auto p : pts) {
			int u_idx = floor((float)p.x / (float)u_step);
			int v_idx = floor((float)p.y / (float)v_step);
			int bin_idx = v_idx * n_bins_u + u_idx;
			if(bin_idx >= 0 && bin_idx < n_bins_total) weight[bin_idx] = 0;
		}
		// std::cout <<" - FEATURE_EXTRACTOR - WeightBin - 'update' : # input points: "<<n_pts << "\n";
	};
};

class FeatureExtractor {
private:
	WeightBin* weight_bin_;

	ParamsORB  params_orb_;
	cv::Ptr<cv::ORB> extractor_orb_; // orb extractor
	// cv::Ptr<cv::Feature2D> extractor_;

private:
	bool  flag_nonmax_;
	bool  flag_debug_;
	int   n_bins_u_;
	int   n_bins_v_;
	int   THRES_FAST_;
	int   NUM_NONMAX_;
	float r_;

public:
	FeatureExtractor();
	~FeatureExtractor();

	void initParams(int n_cols, int n_rows, int n_bins_u, int n_bins_v, int THRES_FAST, int radius);
	void updateWeightBin(const PixelVec& pts);
	void resetWeightBin();
	void suppressCenterBins();
	void extractORBwithBinning(const cv::Mat& img, PixelVec& pts_extracted);
	void extractHarriswithBinning(const cv::Mat& img, PixelVec& pts_extracted);
};

#endif