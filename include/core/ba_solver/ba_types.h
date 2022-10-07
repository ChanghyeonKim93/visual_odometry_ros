#ifndef _BA_TYPES_H_
#define _BA_TYPES_H_

#include <iostream>
#include <vector>

#include <Eigen/Dense>

typedef double _BA_numeric; 

typedef Eigen::Matrix<_BA_numeric,-1,-1> _BA_MatX;

typedef Eigen::Matrix<_BA_numeric,2,2> _BA_Mat22;

typedef Eigen::Matrix<_BA_numeric,2,3> _BA_Mat23;
typedef Eigen::Matrix<_BA_numeric,3,2> _BA_Mat32;

typedef Eigen::Matrix<_BA_numeric,2,6> _BA_Mat26;
typedef Eigen::Matrix<_BA_numeric,6,2> _BA_Mat62;

typedef Eigen::Matrix<_BA_numeric,3,3> _BA_Mat33;

typedef Eigen::Matrix<_BA_numeric,3,6> _BA_Mat36;
typedef Eigen::Matrix<_BA_numeric,6,3> _BA_Mat63;

typedef Eigen::Matrix<_BA_numeric,6,6> _BA_Mat66;

typedef Eigen::Matrix<_BA_numeric,2,1> _BA_Vec2;
typedef Eigen::Matrix<_BA_numeric,3,1> _BA_Vec3;
typedef Eigen::Matrix<_BA_numeric,6,1> _BA_Vec6;

typedef int                            _BA_Index;
typedef Eigen::Matrix<_BA_numeric,2,1> _BA_Pixel;
typedef Eigen::Matrix<_BA_numeric,3,1> _BA_Point;
typedef Eigen::Matrix<_BA_numeric,3,3> _BA_Rot3;
typedef Eigen::Matrix<_BA_numeric,3,1> _BA_Pos3;
typedef Eigen::Matrix<_BA_numeric,4,4> _BA_PoseSE3;
typedef Eigen::Matrix<_BA_numeric,6,1> _BA_PoseSE3Tangent;

typedef std::vector<_BA_Index> _BA_IndexVec;
typedef std::vector<_BA_Pixel> _BA_PixelVec;
typedef std::vector<_BA_Point> _BA_PointVec;

typedef std::vector<_BA_Mat66>              BlockDiagMat66; 
typedef std::vector<_BA_Mat33>              BlockDiagMat33; 
typedef std::vector<std::vector<_BA_Mat63>> BlockFullMat63; 
typedef std::vector<std::vector<_BA_Mat36>> BlockFullMat36;
typedef std::vector<std::vector<_BA_Mat66>> BlockFullMat66;
typedef std::vector<_BA_Vec6>               BlockVec6;
typedef std::vector<_BA_Vec3>               BlockVec3;


#endif