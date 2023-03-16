#ifndef _TYPE_DEFINES_H_
#define _TYPE_DEFINES_H_

#include <iostream> 
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>
#include <Eigen/SparseQR>

#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <ros/ros.h>

class Frame;
class Landmark;
class Camera;

typedef float                     Illumi;
typedef cv::Point2f               Pixel;
typedef Eigen::Vector3f           Point;
typedef bool                      Mask;
typedef std::shared_ptr<Frame>    FramePtr;
typedef std::shared_ptr<Landmark> LandmarkPtr;
typedef std::shared_ptr<Camera>   CameraPtr;

typedef const FramePtr    FrameConstPtr;
typedef const LandmarkPtr LandmarkConstPtr;
typedef const CameraPtr   CameraConstPtr;


typedef std::vector<bool>         BoolVec;
typedef std::vector<int>          IntVec;
typedef std::vector<float>        FloatVec;
typedef std::vector<double>       DoubleVec;

typedef std::vector<Illumi>       IllumiVec;
typedef std::vector<Pixel>        PixelVec;
typedef std::vector<Point>        PointVec;
typedef std::vector<Mask>         MaskVec;
typedef std::vector<FramePtr>     FramePtrVec;
typedef std::vector<LandmarkPtr>  LandmarkPtrVec;

typedef std::vector<cv::Mat>      ImagePyramid;


// For image pyramid
typedef std::vector<IllumiVec>    IllumiVecPyramid;
typedef std::vector<MaskVec>      MaskVecPyramid;

typedef Eigen::Vector3f           Pos3;
typedef Eigen::Matrix3f           Rot3;
typedef Eigen::Matrix4f           PoseSE3;
typedef std::vector<PoseSE3>      PoseSE3Vec;

typedef Eigen::Matrix2f           Mat22;
typedef Eigen::Matrix3f           Mat33;
typedef Eigen::Matrix4f           Mat44;
typedef Eigen::Matrix<float,6,6>  Mat66;

typedef Eigen::Matrix<float,1,5>  Mat15;
typedef Eigen::Matrix<float,5,1>  Mat51;
typedef Eigen::Matrix<float,2,3>  Mat23;
typedef Eigen::Matrix<float,3,2>  Mat32;
typedef Eigen::Matrix<float,2,6>  Mat26;
typedef Eigen::Matrix<float,6,2>  Mat62;
typedef Eigen::Matrix<float,6,3>  Mat63;
typedef Eigen::Matrix<float,3,6>  Mat36;

typedef Eigen::Vector2f           Vec2;
typedef Eigen::Vector3f           Vec3;
typedef Eigen::Vector4f           Vec4;
typedef Eigen::Matrix<float,6,1>  Vec6;

typedef Eigen::Matrix<float,6,1>    PoseSE3Tangent;
typedef std::vector<PoseSE3Tangent> PoseSE3TangentVec;

//For stereo
struct StereoFrame;
class StereoCamera;

typedef std::shared_ptr<StereoFrame>    StereoFramePtr;
typedef const StereoFramePtr            StereoFrameConstPtr;
typedef std::vector<StereoFramePtr>     StereoFramePtrVec;

typedef std::shared_ptr<StereoCamera>   StereoCameraPtr;
typedef const StereoCameraPtr           StereoCameraConstPtr;



// For large matrix (for SFP, depricated)
typedef Eigen::SparseMatrix<float> SpMat;
typedef Eigen::SparseVector<float> SpVec;
typedef Eigen::SparseMatrix<float>::Scalar SpScalar;
typedef Eigen::Triplet<float>      SpTriplet; 
typedef std::vector<SpTriplet>     SpTripletList;

#endif