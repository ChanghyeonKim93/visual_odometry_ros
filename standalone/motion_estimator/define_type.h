#ifndef _DEFINES_TYPE_H_
#define _DEFINES_TYPE_H_

#include <iostream> 
#include <vector>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/SparseCore>
#include <eigen3/Eigen/SparseCholesky>
#include <eigen3/Eigen/SparseLU>
#include <eigen3/Eigen/SparseQR>

#include "opencv4/opencv2/core.hpp"
#include "opencv4/opencv2/core/eigen.hpp"
#include "opencv4/opencv2/imgproc.hpp"
#include "opencv4/opencv2/highgui.hpp"
#include "opencv4/opencv2/calib3d.hpp"

class Frame;
class Landmark;
class Camera;

using Illumi            = float;
using Pixel             = cv::Point2f;
using Point             = Eigen::Vector3f;
using Mask              = bool;
using FramePtr          = std::shared_ptr<Frame>;
using LandmarkPtr       = std::shared_ptr<Landmark>;
using CameraPtr         = std::shared_ptr<Camera>;

using FrameConstPtr     = const FramePtr;
using LandmarkConstPtr  = const LandmarkPtr;
using CameraConstPtr    = const CameraPtr;

using BoolVec           = std::vector<bool>;
using IntVec            = std::vector<int>;
using FloatVec          = std::vector<float>;
using DoubleVec         = std::vector<double>;

using IllumiVec         = std::vector<Illumi>;
using PixelVec          = std::vector<Pixel>;
using PointVec          = std::vector<Point>;
using MaskVec           = std::vector<Mask>;
using FramePtrVec       = std::vector<FramePtr>;
using LandmarkPtrVec    = std::vector<LandmarkPtr>;

using ImagePyramid      = std::vector<cv::Mat>;


// For image pyramid
using IllumiVecPyramid  = std::vector<IllumiVec>;
using MaskVecPyramid    = std::vector<MaskVec>;

using Pos3              = Eigen::Vector3f;
using Rot3              = Eigen::Matrix3f;
using PoseSE3           = Eigen::Matrix4f;
using PoseSE3Vec        = std::vector<PoseSE3>;

using Mat22             = Eigen::Matrix2f;
using Mat33             = Eigen::Matrix3f;
using Mat44             = Eigen::Matrix4f;
using Mat66             = Eigen::Matrix<float,6,6>;

using Mat15             = Eigen::Matrix<float,1,5>;
using Mat51             = Eigen::Matrix<float,5,1>;
using Mat23             = Eigen::Matrix<float,2,3>;
using Mat32             = Eigen::Matrix<float,3,2>;
using Mat26             = Eigen::Matrix<float,2,6>;
using Mat62             = Eigen::Matrix<float,6,2>;
using Mat63             = Eigen::Matrix<float,6,3>;
using Mat36             = Eigen::Matrix<float,3,6>;

using Vec2              = Eigen::Vector2f;
using Vec3              = Eigen::Vector3f;
using Vec4              = Eigen::Vector4f;
using Vec5              = Eigen::Matrix<float,5,1>;
using Vec6              = Eigen::Matrix<float,6,1>;

//For stereo
struct StereoFrame;
class StereoCamera;

using StereoFramePtr       = std::shared_ptr<StereoFrame>;
using StereoFrameConstPtr  = const StereoFramePtr;
using StereoFramePtrVec    = std::vector<StereoFramePtr>;

using StereoCameraPtr      = std::shared_ptr<StereoCamera>;
using StereoCameraConstPtr = const StereoCameraPtr;

#endif