#ifndef _GEOMETRY_LIBRARY_H_
#define _GEOMETRY_LIBRARY_H_

#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;
namespace geometry {
    Matrix3d skewMat(const Vector3d& v);
    Matrix3f skewMat_f(const Vector3f& v);

    Matrix4d q_right_mult(const Vector4d& q);
    Matrix4f q_right_mult_f(const Vector4f& q);

    Matrix4d q_left_mult(const Vector4d& q);
    Matrix4f q_left_mult_f(const Vector4f& q);

    Vector4d q_conj(const Vector4d& q);
    Vector4f q_conj_f(const Vector4f& q);

    Vector4d q1_mult_q2(const Vector4d& q1, const Vector4d& q2);
    Vector4f q1_mult_q2_f(const Vector4f& q1, const Vector4f& q2);

    Matrix3d q2r(const Vector4d& q);
    Matrix3f q2r_f(const Vector4f& q);

    Vector4d rotvec2q(const Vector3d& w);
    Vector4f rotvec2q_f(const Vector3f& w);

    Matrix3d a2r(double r, double p, double y);
    Matrix3f a2r_f(float r, float p, float y);

    Vector4d r2q(const Matrix3d& R);
    Vector4f r2q_f(const Matrix3f& R);

    void se3Exp_f(const Eigen::Matrix<float,6,1>& xi, Eigen::Matrix4f& T);
    void SE3Log_f(const Eigen::Matrix<float,4,4>& T, Eigen::Matrix<float,6,1>& xi);
};


#endif