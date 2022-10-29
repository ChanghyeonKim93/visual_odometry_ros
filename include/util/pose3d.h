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

#define EPSILON_FLOAT 1e-5f;


class Rotation3
{
private:
    SO3 R_; // SO3
    Quaternion4 q_; // w x y z (normalized quaternion, ||q||_2 = 1)

public:
    Rotation3()
    {
        R_= SO3::Identity();
        q_ << 1.0f, 0.0f, 0.0f, 0.0f;
    };

    // Copy constructor
    Rotation3(const Rotation3& Rotation3);
    ~Rotation3() { };

// Get methods
public:
    const SO3&         R() const { return R_; };
    const Quaternion4& q() const { return q_; };
    const SO3&         inverse() { return R_.transpose(); };

    _mat_numeric determinant() 
    {
        _mat_numeric val = 0.0;
        val += R_(0,0)*( R_(1,1)*R_(2,2) - R_(1,2)*R_(2,1));
        val -= R_(0,1)*( R_(1,0)*R_(2,2) - R_(1,2)*R_(2,0));
        val += R_(0,2)*( R_(1,0)*R_(2,1) - R_(1,1)*R_(2,0));
        return val;
    };

// Set method
public:
    void setIdentity() 
    { 
        R_ = SO3::Identity(); 
        q_ << 1.0f, 0.0f, 0.0f, 0.0f;
    };

// Operator overloading
public:
    Rotation3& operator=(const Rotation3& Rotation3)
    {
        R_ << Rotation3.R();
        q_ << Rotation3.q();
        
        // Normalize and project SO(3) space
        q_.normalize();



        return *this;
    };

    // 곱셈 연산자.
    Rotation3 operator * (Rotation3& rot)
    {
        Rotation3 rot_new;
        
        return rot_new;
    };

    friend std::ostream& operator << (std::ostream& os, const Rotation3& rot)
    {
        os << rot.R_;
        return os;
    };

private:
    void convertRotationToQuaternion()
    {
        _mat_numeric m00 = R_(0,0);
        _mat_numeric m11 = R_(1,1);
        _mat_numeric m22 = R_(2,2);

        _mat_numeric m21 = R_(2,1);
        _mat_numeric m12 = R_(1,2);
        _mat_numeric m02 = R_(0,2);
        _mat_numeric m20 = R_(2,0);
        _mat_numeric m10 = R_(1,0);
        _mat_numeric m01 = R_(0,1);

        /*
             To resolve the signs, find the largest of q0, q1, q2, q3 and 
            assume its sign is positive. 
             Then compute the remaining components as shown in the table below.
             Taking the largest magnitude avoids division by small numbers, 
            which would reduce numerical accuracy. 
            https://danceswithcode.net/engineeringnotes/quaternions/quaternions.html

        */

        _mat_numeric tr = m00 + m11 + m22;

        if( tr > 0)
        {

        }
        else if(1)
        {

        }
        else if(1)
        {

        }
        else
        {

        }
    };

    void convertQuaternionToRotation()
    {
        const _mat_numeric& w = q_(0);
        const _mat_numeric& x = q_(1);
        const _mat_numeric& y = q_(2);
        const _mat_numeric& z = q_(3);

        _mat_numeric w2 = w*w;
        _mat_numeric x2 = x*x;
        _mat_numeric y2 = y*y;
        _mat_numeric z2 = z*z;

        _mat_numeric xy = x*y;
        _mat_numeric wz = w*z;
        _mat_numeric xz = x*z;
        _mat_numeric wy = w*y;
        _mat_numeric wx = w*x;
        _mat_numeric yz = y*z;

        R_ <<  w2+x2-y2-z2, 2.0*(xy-wz), 2.0*(xz+wy),
               2.0*(xy+wz), w2-x2+y2-z2, 2.0*(yz-wx),
               2.0*(xz-wy), 2.0*(yz+wx), w2-x2-y2+z2;    
    };

    void q_mult_back(const Quaternion4& q2)
    {
        const Quaternion4& q1 = q_;
        q_ << 
        q1(0)*q2(0)-q1(1)*q2(1)-q1(2)*q2(2)-q1(3)*q2(3),
        q1(0)*q2(1)+q1(1)*q2(0)+q1(2)*q2(3)-q1(3)*q2(2),
        q1(0)*q2(2)-q1(1)*q2(3)+q1(2)*q2(0)+q1(3)*q2(1),
        q1(0)*q2(3)+q1(1)*q2(2)-q1(2)*q2(1)+q1(3)*q2(0);
    };

    void q_mult_front(const Quaternion4& q1)
    {
        const Quaternion4& q2 = q_;
        q_ << 
        q1(0)*q2(0)-q1(1)*q2(1)-q1(2)*q2(2)-q1(3)*q2(3),
        q1(0)*q2(1)+q1(1)*q2(0)+q1(2)*q2(3)-q1(3)*q2(2),
        q1(0)*q2(2)-q1(1)*q2(3)+q1(2)*q2(0)+q1(3)*q2(1),
        q1(0)*q2(3)+q1(1)*q2(2)-q1(2)*q2(1)+q1(3)*q2(0);
    }
};




class Pose3D
{
public:

private:
    Rotation3 rot_; // SO3
    Position3 t_; // 3d translation vector
    
    se3 xi_;
    SE3 T_;

public:
    Pose3D() : rot_()
    {
        t_.setZero();
        xi_.setZero();
        T_.setZero();
    };
    
    ~Pose3D() { };

// Get method (const)
public:
    const SO3& R() const         { return rot_.R(); };
    const Position3& t() const   { return t_; };
    const Quaternion4& q() const { return rot_.q(); };
    


private:


};

#endif