#ifndef _DEFINE_BA_TYPE_H_
#define _DEFINE_BA_TYPE_H_

#include <iostream>
#include <vector>
#include <Eigen/Dense>

typedef double _BA_Numeric; 

typedef Eigen::Matrix<_BA_Numeric,-1,-1>   _BA_MatX;

typedef _BA_Numeric                        _BA_Mat11;

typedef Eigen::Matrix<_BA_Numeric,2,2>     _BA_Mat22;

typedef Eigen::Matrix<_BA_Numeric,1,3>     _BA_Mat13;
typedef Eigen::Matrix<_BA_Numeric,3,1>     _BA_Mat31;

typedef Eigen::Matrix<_BA_Numeric,2,3>     _BA_Mat23;
typedef Eigen::Matrix<_BA_Numeric,3,2>     _BA_Mat32;

typedef Eigen::Matrix<_BA_Numeric,2,6>     _BA_Mat26;
typedef Eigen::Matrix<_BA_Numeric,6,2>     _BA_Mat62;

typedef Eigen::Matrix<_BA_Numeric,3,3>     _BA_Mat33;

typedef Eigen::Matrix<_BA_Numeric,3,6>     _BA_Mat36;
typedef Eigen::Matrix<_BA_Numeric,6,3>     _BA_Mat63;

typedef Eigen::Matrix<_BA_Numeric,6,6>     _BA_Mat66;

typedef _BA_Numeric                        _BA_Vec1;
typedef Eigen::Matrix<_BA_Numeric,2,1>     _BA_Vec2;
typedef Eigen::Matrix<_BA_Numeric,3,1>     _BA_Vec3;
typedef Eigen::Matrix<_BA_Numeric,6,1>     _BA_Vec6;

typedef int                                _BA_Index;
typedef Eigen::Matrix<_BA_Numeric,2,1>     _BA_Pixel;
typedef Eigen::Matrix<_BA_Numeric,3,1>     _BA_Point;
typedef Eigen::Matrix<_BA_Numeric,3,3>     _BA_Rot3;
typedef Eigen::Matrix<_BA_Numeric,3,1>     _BA_Pos3;
typedef Eigen::Matrix<_BA_Numeric,4,4>     _BA_PoseSE3;
typedef Eigen::Matrix<_BA_Numeric,6,1>     _BA_PoseSE3Tangent;

typedef std::vector<_BA_Numeric>           _BA_ErrorVec;
typedef std::vector<_BA_Index>             _BA_IndexVec;
typedef std::vector<_BA_Pixel>             _BA_PixelVec;
typedef std::vector<_BA_Point>             _BA_PointVec;

typedef std::vector<_BA_Mat33>              DiagBlockMat33; 
typedef std::vector<_BA_Mat66>              DiagBlockMat66; 

typedef std::vector<std::vector<_BA_Mat11>> FullBlockMat11;
typedef std::vector<std::vector<_BA_Mat13>> FullBlockMat13;
typedef std::vector<std::vector<_BA_Mat31>> FullBlockMat31;
typedef std::vector<std::vector<_BA_Mat33>> FullBlockMat33; 
typedef std::vector<std::vector<_BA_Mat63>> FullBlockMat63; 
typedef std::vector<std::vector<_BA_Mat36>> FullBlockMat36;
typedef std::vector<std::vector<_BA_Mat66>> FullBlockMat66;

typedef std::vector<_BA_Vec1>               BlockVec1;
typedef std::vector<_BA_Vec3>               BlockVec3;
typedef std::vector<_BA_Vec6>               BlockVec6;


#endif