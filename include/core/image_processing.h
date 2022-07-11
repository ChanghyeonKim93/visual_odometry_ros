#ifndef _IMAGE_PROCESSING_H_
#define _IMAGE_PROCESSING_H_
#include <iostream>
#include <string>
#include <opencv2/core.hpp>
#include "core/type_defines.h"

namespace image_processing {
	std::string type2str(cv::Mat img);
};
#endif