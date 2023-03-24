#ifndef _POSE3D_H_
#define _POSE3D_H_

#include <iostream>
#include <vector>

#include <Eigen/Dense>

typedef float _mat_numeric;

typedef Eigen::Matrix<_mat_numeric,3,1> Position3;

typedef Eigen::Matrix<_mat_numeric,4,1> Quaternion4;

typedef Eigen::Matrix<_mat_numeric,3,1> so3;
typedef Eigen::Matrix<_mat_numeric,3,3> SO3;

typedef Eigen::Matrix<_mat_numeric,6,1> se3;
typedef Eigen::Matrix<_mat_numeric,4,4> SE3;


#define FLT_EPSILON 1.1920292896e-07f // smallest such that 1.0+FLT_EPSILON != 1.0
#define DBL_EPSILON 2.2204460492503131e-016 // smallest such that 1.0 + DBL_EPSILON != 1.0

// Rotation inintialization:
// rotvec to R : w -> R -> q
// R to q : R -> q
// q to R : q -> R
// multiplication: R -> q and q1*q2
// multiplication by rotvec: w -> R -> q and q1*q2

class Rotation3;
class Pose3D;

class Rotation3
{
private:
    SO3 R_; // SO3
    Quaternion4 q_; // w x y z (normalized quaternion, ||q||_2 = 1)
    so3 w_;

public:
    Rotation3();
    Rotation3(const SO3& R);
    Rotation3(const so3& w);
    Rotation3(const Quaternion4& q);
    // Copy constructor
    Rotation3(const Rotation3& rot);

// Get methods
public:
    const SO3&         R() const;
    const Quaternion4& q() const;
    const so3&         w() const;

    Rotation3          inverse() const;

    _mat_numeric determinant() const;

// Set method
public:
    void setIdentity();

private:
    void initByRotation(const SO3& R);
    void initByQuaternion(const Quaternion4& q);
    void initByAxisAngle(const so3& w);

// Operator overloading
// 대입 연산 (깊은 복사)
// 곱셈 연산
// 곱셈 대입 연산 (자기 자신에게 곱셈하고 자기 자신을 리턴)
public:
    Rotation3& operator  = (const Rotation3& Rotation3); // 대입 복사 연산자.
    Rotation3  operator  * (const Rotation3& rot) const; // 곱셈 연산자.
    Rotation3& operator *= (const Rotation3& rot); // 곱셈 대입 연산자.

    // 대입 연산자.
    void operator<<(const Rotation3& rot);
    void operator<<(const SO3& R);

    void operator<<(const Quaternion4& q);
    void operator<<(const so3& w);
    Pose3D operator,(const Position3& t);

    friend std::ostream& operator << (std::ostream& os, const Rotation3& rot);

private:
    void convertRotationToQuaternion();
    void convertQuaternionToRotation();
    void convertAxisAngleToQuaternion();
    void convertQuaternionToAxisAngle();

    Quaternion4 q1_mult_q2(const Quaternion4& q1, const Quaternion4& q2) const;
};


class Pose3D
{
private:
    Rotation3 rot_; // SO3
    Position3 t_; // 3d translation vector
    
    SE3 T_;

public:
    Pose3D();
    Pose3D(const Rotation3& rot, const Position3& t);
    Pose3D(const SO3& R, const Position3& t);
    Pose3D(const Quaternion4& q, const Position3& t);
    Pose3D(const so3& w, const Position3& t);
    Pose3D(const Pose3D& pose);
    

// Get method (const)
public:
    const Rotation3& rotation() const;
    const Quaternion4&      q() const;

    const SO3&       R() const;
    const Position3& t() const;
    const SE3&       T() const;
    
    Pose3D inverse() const;



// Set methods
public:
    void setIdentity();

private:
    void initByRotationAndTranslation(const SO3& R, const Position3& t);
    void initByQuaternionAndTranslation(const Quaternion4& q, const Position3& t);
    void initByAxisAngleAndTranslation(const so3& w, const Position3& t);


// Operator overloading
public:
    Pose3D& operator=(const Pose3D& pose);
    // 곱셈 연산자
    Pose3D operator*(const Pose3D& pose) const;
    // 곱셈 대입 연산자
    Pose3D& operator*=(const Pose3D& pose);
    // 대입 연산자.
    void operator<<(const Pose3D& pose);
    friend std::ostream& operator << (std::ostream& os, const Pose3D& pose);
};

#endif