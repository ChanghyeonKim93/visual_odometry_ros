#include "dataset_loader.h"

void dataset_loader::getImageFileNames_KITTI(
	const std::string& dir_odom_dataset, const std::string& num_dataset,
	DatasetStruct& dataset)
{
	// load KITTI odometry dataset.
	std::vector<double>& timestamps          = dataset.timestamps;
	std::vector<std::string>& img_file_names = dataset.image_names;

	std::string str;
	uint32_t& n_data = dataset.n_data;
	n_data = 0;

	// get timestamps
	std::string dir_specific_dataset; 
	dir_specific_dataset = dir_odom_dataset + "/dataset/sequences/" + num_dataset + "/";

	std::ifstream timestamp_file;
	timestamp_file.open(std::string(dir_specific_dataset + "times.txt"));
	if(!timestamp_file.is_open())
		throw std::runtime_error("timestamp file cannot be open!\n");

	while (!timestamp_file.eof()){
		char arr[256];
		timestamp_file.getline(arr, 256);
		timestamps.push_back(std::stod(std::string(arr)));
		++n_data;
	}
	timestamp_file.close();

	// make filenames
	img_file_names.resize(n_data);
	for (uint32_t i = 0; i < n_data; ++i) {
		std::stringstream ss;
		ss << std::setw(6) << std::setfill('0') << i;
		img_file_names[i] = dir_specific_dataset + "image_0/" + ss.str() + ".png";
	}

	dataset.duration = timestamps.back() - timestamps.front();
};