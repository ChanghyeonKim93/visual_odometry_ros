#ifndef _DEFINE_BA_TYPE_H_
#define _DEFINE_BA_TYPE_H_

#include <iostream>
#include <vector>

#include <eigen3/Eigen/Dense>

using _BA_Numeric = double; 

using _BA_MatX = Eigen::Matrix<_BA_Numeric,-1,-1>;

using _BA_Mat11 = _BA_Numeric;

using _BA_Mat22 = Eigen::Matrix<_BA_Numeric,2,2>;

using _BA_Mat13 = Eigen::Matrix<_BA_Numeric,1,3>;
using _BA_Mat31 = Eigen::Matrix<_BA_Numeric,3,1>;

using _BA_Mat23 = Eigen::Matrix<_BA_Numeric,2,3>;
using _BA_Mat32 = Eigen::Matrix<_BA_Numeric,3,2>;

using _BA_Mat26 = Eigen::Matrix<_BA_Numeric,2,6>;
using _BA_Mat62 = Eigen::Matrix<_BA_Numeric,6,2>;

using _BA_Mat33 = Eigen::Matrix<_BA_Numeric,3,3>;

using _BA_Mat36 = Eigen::Matrix<_BA_Numeric,3,6>;
using _BA_Mat63 = Eigen::Matrix<_BA_Numeric,6,3>;

using _BA_Mat66 = Eigen::Matrix<_BA_Numeric,6,6>;

using _BA_Vec1 = _BA_Numeric;
using _BA_Vec2 = Eigen::Matrix<_BA_Numeric,2,1>;
using _BA_Vec3 = Eigen::Matrix<_BA_Numeric,3,1>;
using _BA_Vec6 = Eigen::Matrix<_BA_Numeric,6,1>;

using _BA_Index = int;
using _BA_Pixel = Eigen::Matrix<_BA_Numeric,2,1>;
using _BA_Point = Eigen::Matrix<_BA_Numeric,3,1>;
using _BA_Rot3 = Eigen::Matrix<_BA_Numeric,3,3>;
using _BA_Pos3 = Eigen::Matrix<_BA_Numeric,3,1>;
using _BA_PoseSE3 = Eigen::Matrix<_BA_Numeric,4,4>;
using _BA_PoseSE3Tangent = Eigen::Matrix<_BA_Numeric,6,1>;

using _BA_ErrorVec = std::vector<_BA_Numeric>;
using _BA_IndexVec = std::vector<_BA_Index>;
using _BA_PixelVec = std::vector<_BA_Pixel>;
using _BA_PointVec = std::vector<_BA_Point>;

using DiagBlockMat33 = std::vector<_BA_Mat33>; 
using DiagBlockMat66 = std::vector<_BA_Mat66>; 

using FullBlockMat11 = std::vector<std::vector<_BA_Mat11>>;
using FullBlockMat13 = std::vector<std::vector<_BA_Mat13>>;
using FullBlockMat31 = std::vector<std::vector<_BA_Mat31>>;
using FullBlockMat33 = std::vector<std::vector<_BA_Mat33>>; 
using FullBlockMat63 = std::vector<std::vector<_BA_Mat63>>; 
using FullBlockMat36 = std::vector<std::vector<_BA_Mat36>>;
using FullBlockMat66 = std::vector<std::vector<_BA_Mat66>>;

using BlockVec1 = std::vector<_BA_Vec1>;
using BlockVec3 = std::vector<_BA_Vec3>;
using BlockVec6 = std::vector<_BA_Vec6>;

#endif