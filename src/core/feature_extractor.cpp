#include "core/feature_extractor.h"
#include <algorithm>

FeatureExtractor::FeatureExtractor() :params_orb_() {
	weight_bin_  = nullptr;
	flag_debug_  = false;
	flag_nonmax_ = true;

	n_bins_u_   = 0;
	n_bins_v_   = 0;

	THRES_FAST_ = 0;
	r_          = 0;

	NUM_NONMAX_ = 0;

	std::cout << " - FEATURE_EXTRACTOR is constructed.\n";
};
FeatureExtractor::~FeatureExtractor() {
	if (weight_bin_ != nullptr) delete weight_bin_;

	std::cout << " - FEATURE_EXTRACTOR is deleted.\n";
};

void FeatureExtractor::initParams(int n_cols, int n_rows, int n_bins_u, int n_bins_v, int THRES_FAST, int radius) {
	std::cout << " - FEATURE_EXTRACTOR - 'initParams'\n";

	this->extractor_orb_ = cv::ORB::create();

	std::cout << " ORB is created.\n";

	this->flag_debug_ = false;
	this->flag_nonmax_ = true;

	this->NUM_NONMAX_ = 5;

	this->n_bins_u_ = n_bins_u;
	this->n_bins_v_ = n_bins_v;

	this->THRES_FAST_ = THRES_FAST;
	this->r_ = radius;

	this->params_orb_.FastThreshold = this->THRES_FAST_;
	this->params_orb_.n_bins_u = this->n_bins_u_;
	this->params_orb_.n_bins_v = this->n_bins_v_;

	extractor_orb_->setMaxFeatures(100);
	extractor_orb_->setScaleFactor(1.05);
	extractor_orb_->setNLevels(1);
	extractor_orb_->setEdgeThreshold(6);
	extractor_orb_->setFirstLevel(0);
	extractor_orb_->setWTA_K(2);
	extractor_orb_->setScoreType(cv::ORB::FAST_SCORE);
	extractor_orb_->setPatchSize(31);
	extractor_orb_->setFastThreshold(this->THRES_FAST_);

	this->weight_bin_ = new WeightBin();
	this->weight_bin_->init(n_cols, n_rows, this->n_bins_u_, this->n_bins_v_);
};

void FeatureExtractor::resetWeightBin() {
	// printf(" - FEATURE_EXTRACTOR - 'resetWeightBin'\n");

	weight_bin_->reset();
};

void FeatureExtractor::suppressCenterBins(){
	int u_cent = params_orb_.n_bins_u/2;	
	int v_cent = params_orb_.n_bins_v/2;


	int win_sz_u = (int)(0.3f*params_orb_.n_bins_u);
	int win_sz_v = (int)(0.3f*params_orb_.n_bins_v);
	for(int w = -win_sz_v; w <= win_sz_v; ++w){
		int v_idx = params_orb_.n_bins_u*(w+v_cent);
		for(int u = -win_sz_u; u <= win_sz_u; ++u){
			int bin_idx = v_idx + u;
			// std::cout << bin_idx << std::endl;
			weight_bin_->weight[bin_idx] = 0;
		}
	}
};

void FeatureExtractor::updateWeightBin(const PixelVec& fts) {
	// std::cout << " - FEATURE_EXTRACTOR - 'updateWeightBin'\n";

	weight_bin_->reset();
	weight_bin_->update(fts);
};

void FeatureExtractor::extractORBwithBinning(const cv::Mat& img, PixelVec& pts_extracted) {
	// INPUT IMAGE MUST BE CV_8UC1 image.

	cv::Mat img_in;
	if (img.type() != CV_8UC1) {
		img.convertTo(img_in, CV_8UC1);
		std::cout << "in extractor, image is converted to the CV_8UC1.\n";
	}
	else img_in = img;


	int overlap = floor(1 * params_orb_.EdgeThreshold);

	std::vector<cv::KeyPoint> fts_tmp;

	pts_extracted.resize(0);
	pts_extracted.reserve(1000);
	fts_tmp.reserve(100);

	int*& u_idx = weight_bin_->u_bound;
	int*& v_idx = weight_bin_->v_bound;

	int v_range[2] = { 0,0 };
	int u_range[2] = { 0,0 };
	for (int v = 0; v < n_bins_v_; ++v) {
		for (int u = 0; u < n_bins_u_; ++u) {
			int bin_idx = v * n_bins_u_ + u;
			int n_pts_desired = weight_bin_->weight[bin_idx] * params_orb_.MaxFeatures;
			if (n_pts_desired > 0) {
				// Set the maximum # of features for this bin.
				// Crop a binned image
				if (v == 0) {
					v_range[0] = v_idx[v];
					v_range[1] = v_idx[v + 1] + overlap;
				}
				else if (v == n_bins_v_ - 1) {
					v_range[0] = v_idx[v] - overlap;
					v_range[1] = v_idx[v + 1];
				}
				else {
					v_range[0] = v_idx[v] - overlap;
					v_range[1] = v_idx[v + 1] + overlap;
				}

				if (u == 0) {
					u_range[0] = u_idx[u];
					u_range[1] = u_idx[u + 1] + overlap;
				}
				else if (u == n_bins_u_ - 1) {
					u_range[0] = u_idx[u] - overlap;
					u_range[1] = u_idx[u + 1];
				}
				else {
					u_range[0] = u_idx[u] - overlap;
					u_range[1] = u_idx[u + 1] + overlap;
				}
				// image sampling
				// TODO: which one is better? : sampling vs. masking
				// std::cout << "set ROI \n";
				cv::Rect roi = cv::Rect(cv::Point(u_range[0], v_range[0]), cv::Point(u_range[1], v_range[1]));
				// std::cout << "roi: " << roi << std::endl;
				// std::cout << "image size : " << img_in.size() <<std::endl;
				
				cv::Mat img_small = img_in(roi);
				fts_tmp.resize(0);
				extractor_orb_->detect(img_small, fts_tmp);

				int n_pts_tmp = fts_tmp.size();
				if (n_pts_tmp > 0) { //feature can be extracted from this bin.
					int u_offset = 0; int v_offset = 0;
					if (v == 0) v_offset = 0;
					else if (v == n_bins_v_ - 1) v_offset = v_idx[v] - overlap - 1;
					else v_offset = v_idx[v] - overlap - 1;
					if (u == 0) u_offset = 0;
					else if (u == n_bins_u_ - 1) u_offset = u_idx[u] - overlap - 1;
					else u_offset = u_idx[u] - overlap - 1;

					std::sort(fts_tmp.begin(), fts_tmp.end(), [](const cv::KeyPoint &a, const cv::KeyPoint &b) { return a.response > b.response; });
					if (flag_nonmax_ == true && fts_tmp.size() > NUM_NONMAX_) { // select most responsive point in a bin
						fts_tmp.resize(NUM_NONMAX_); // maximum two points.
					}
					else fts_tmp.resize(1);

					cv::Point2f pt_offset(u_offset, v_offset);
					for (auto it : fts_tmp) {
						it.pt += pt_offset;
						pts_extracted.push_back(it.pt);
					}
				}
			}
		}
	}

	// Final result
	// std::cout << " - FEATURE_EXTRACTOR - 'extractORBwithBinning' - # detected pts : " << pts_extracted.size() << std::endl;
};

void FeatureExtractor::extractHarriswithBinning(const cv::Mat& img, PixelVec& pts_extracted){
	// INPUT IMAGE MUST BE CV_8UC1 image.

	cv::Mat img_in;
	if (img.type() != CV_8UC1) {
		img.convertTo(img_in, CV_8UC1);
	}

	int overlap = floor(1 * params_orb_.EdgeThreshold);

	std::vector<cv::KeyPoint> fts_tmp;

	pts_extracted.resize(0);
	pts_extracted.reserve(1000);
	fts_tmp.reserve(100);

	int*& u_idx = weight_bin_->u_bound;
	int*& v_idx = weight_bin_->v_bound;

	int v_range[2] = { 0,0 };
	int u_range[2] = { 0,0 };
	for (int v = 0; v < n_bins_v_; ++v) {
		for (int u = 0; u < n_bins_u_; ++u) {
			int bin_idx = v * n_bins_u_ + u;
			int n_pts_desired = weight_bin_->weight[bin_idx] * params_orb_.MaxFeatures;
			if (n_pts_desired > 0) {
				// Set the maximum # of features for this bin.
				// Crop a binned image
				if (v == 0) {
					v_range[0] = v_idx[v];
					v_range[1] = v_idx[v + 1] + overlap;
				}
				else if (v == n_bins_v_ - 1) {
					v_range[0] = v_idx[v] - overlap;
					v_range[1] = v_idx[v + 1];
				}
				else {
					v_range[0] = v_idx[v] - overlap;
					v_range[1] = v_idx[v + 1] + overlap;
				}

				if (u == 0) {
					u_range[0] = u_idx[u];
					u_range[1] = u_idx[u + 1] + overlap;
				}
				else if (u == n_bins_u_ - 1) {
					u_range[0] = u_idx[u] - overlap;
					u_range[1] = u_idx[u + 1];
				}
				else {
					u_range[0] = u_idx[u] - overlap;
					u_range[1] = u_idx[u + 1] + overlap;
				}

				// image sampling
				// TODO: which one is better? : sampling vs. masking
				cv::Rect roi = cv::Rect(cv::Point(u_range[0], v_range[0]), cv::Point(u_range[1], v_range[1]));

				// extractor_orb_->detect(img(roi), fts_tmp);
				double quality_level = 0.07;
				int max_corners      = 100;
				double min_px_dist   = 5.0; 
				double k = 0.0005;
				std::vector<cv::Point2f> pts_tmp;
				cv::goodFeaturesToTrack(img_in(roi), pts_tmp, max_corners, quality_level, min_px_dist, {}, 5, true, k);

				fts_tmp.resize(0);
				for(auto it : pts_tmp){
					fts_tmp.emplace_back(it,1);
				}
			
				int n_pts_tmp = fts_tmp.size();
				if (n_pts_tmp > 0) { //feature can be extracted from this bin.
					int u_offset = 0; int v_offset = 0;
					if (v == 0) v_offset = 0;
					else if (v == n_bins_v_ - 1) v_offset = v_idx[v] - overlap - 1;
					else v_offset = v_idx[v] - overlap - 1;
					if (u == 0) u_offset = 0;
					else if (u == n_bins_u_ - 1) u_offset = u_idx[u] - overlap - 1;
					else u_offset = u_idx[u] - overlap - 1;

					std::sort(fts_tmp.begin(), fts_tmp.end(), [](const cv::KeyPoint &a, const cv::KeyPoint &b) { return a.response > b.response; });
					if (flag_nonmax_ == true && fts_tmp.size() > NUM_NONMAX_) { // select most responsive point in a bin
						fts_tmp.resize(NUM_NONMAX_); // maximum two points.
					}
					else fts_tmp.resize(1);

					cv::Point2f pt_offset(u_offset, v_offset);
					for (auto it : fts_tmp) {
						it.pt += pt_offset;
						pts_extracted.push_back(it.pt);
					}
				}
			}
		}
	}

	// std::cout << " - FEATURE_EXTRACTOR - 'extractORBwithBinning' - # detected pts : " << pts_extracted.size() << std::endl;
};