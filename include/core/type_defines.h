#ifndef _TYPE_DEFINES_H_
#define _TYPE_DEFINES_H_

#include <iostream> 
#include <vector>

#include <Eigen/Dense>

#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <ros/ros.h>

class Frame;
class Landmark;

typedef cv::Point2f               Pixel;
typedef Eigen::Vector3f           Point;
typedef bool                      Mask;
typedef std::shared_ptr<Frame>    FramePtr;
typedef std::shared_ptr<Landmark> LandmarkPtr;

typedef std::vector<Pixel>        PixelVec;
typedef std::vector<Point>        PointVec;
typedef std::vector<Mask>         MaskVec;
typedef std::vector<FramePtr>     FramePtrVec;
typedef std::vector<LandmarkPtr>  LandmarkPtrVec;

#endif