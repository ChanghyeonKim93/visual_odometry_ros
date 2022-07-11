#ifndef _GEOMETRY_LIBRARY_H_
#define _GEOMETRY_LIBRARY_H_

#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;
namespace geometry {
    Matrix3d skewMat(const Vector3d& v);
    Matrix4d q_right_mult(const Vector4d& q);
    Matrix4d q_left_mult(const Vector4d& q);
    Vector4d q_conj(const Vector4d& q);
    Vector4d q1_mult_q2(const Vector4d& q1, const Vector4d& q2);
    Matrix3d q2r(const Vector4d& q);
    Vector4d rotvec2q(const Vector3d& w);
    Matrix3d a2r(double r, double p, double y);
    Vector4d r2q(const Matrix3d& R);
};


#endif