#include "feature_extractor.h"
#include <algorithm>

FeatureExtractor::FeatureExtractor() {
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

	this->extractor_ = cv::ORB::create();

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

	extractor_->setMaxFeatures(100);
	extractor_->setScaleFactor(1.05);
	extractor_->setNLevels(1);
	extractor_->setEdgeThreshold(6);
	extractor_->setFirstLevel(0);
	extractor_->setWTA_K(2);
	extractor_->setScoreType(cv::ORB::FAST_SCORE);
	extractor_->setPatchSize(31);
	extractor_->setFastThreshold(this->THRES_FAST_);

	this->weight_bin_ = new WeightBin();
	this->weight_bin_->init(n_cols, n_rows, this->n_bins_u_, this->n_bins_v_);
};

void FeatureExtractor::resetWeightBin() {
	printf(" - FEATURE_EXTRACTOR - 'resetWeightBin'\n");

	weight_bin_->reset();
};

void FeatureExtractor::updateWeightBin(const Pixels& pts) {
	printf(" - FEATURE_EXTRACTOR - 'updateWeightBin'\n");

	weight_bin_->reset();
	weight_bin_->update(pts);
};

void FeatureExtractor::extractORBwithBinning(const cv::Mat& img, Pixels& pts_extracted) {
	// INPUT IMAGE MUST BE CV_8UC1 image.

	cv::Mat img_in;
	if (img.type() != CV_8UC1) {
		img.convertTo(img_in, CV_8UC1);
	}

	pts_extracted.resize(0);

	int overlap = floor(1 * params_orb_.EdgeThreshold);

	std::vector<cv::KeyPoint> pts_kp;
	Features fts_tmp;

	pts_kp.reserve(10000);
	fts_tmp.reserve(3000);

	cv::Mat mask(0 * cv::Mat::ones(img.size(), CV_8UC1));

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
				cv::Rect rect = cv::Rect(cv::Point(u_range[0], v_range[0]), cv::Point(u_range[1], v_range[1]));
				cv::Mat  img_bin = img(rect);

				fts_tmp.resize(0);
				extractor_->detect(img_bin, fts_tmp);

				int n_pts_tmp = fts_tmp.size();
				if (n_pts_tmp > 0) { //feature can be extracted from this bin.
					int u_offset = 0; int v_offset = 0;
					if (v == 0) v_offset = 0;
					else if (v == n_bins_v_ - 1) v_offset = v_idx[v] - overlap - 1;
					else v_offset = v_idx[v] - overlap - 1;
					if (u == 0) u_offset = 0;
					else if (u == n_bins_u_ - 1) u_offset = u_idx[u] - overlap - 1;
					else u_offset = u_idx[u] - overlap - 1;

					std::sort(fts_tmp.begin(), fts_tmp.end(), [](const Feature &a, const Feature &b) { return a.response > b.response; });
					if (flag_nonmax_ == true && fts_tmp.size() > NUM_NONMAX_) { // select most responsive point in a bin
						fts_tmp.resize(NUM_NONMAX_); // maximum two points.
					}
					else fts_tmp.resize(1);

					cv::Point2f pt_offset(u_offset, v_offset);
					for (auto it : fts_tmp) {
						it.pt += pt_offset;
						pts_kp.push_back(it);
					}
				}
			}
		}
	}

	// Final result
	if (pts_kp.size() > 0) {
		for (auto it : pts_kp) pts_extracted.emplace_back(it.pt);
	}

	std::cout << " - FEATURE_EXTRACTOR - 'extractORBwithBinning' - # detected pts : " << pts_kp.size() << std::endl;
};