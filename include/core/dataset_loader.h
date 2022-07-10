#ifndef _DATASET_LOAD_H_
#define _DATASET_LOAD_H_
#include <iostream>
#include <string>
#include <cstring>
#include <vector>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace dataset_loader {
	struct DatasetStruct {
		std::vector<double>      timestamps;
		std::vector<std::string> image_names;

		// Statistics
		uint32_t n_data;
		double duration;
		double fps;

		DatasetStruct() : n_data(0), duration(0), fps(0) {};

	};
	
	void getImageFileNames_KITTI(
		const std::string& dir_dataset, const std::string& num_dataset,
		DatasetStruct& dataset);
	
};


#endif