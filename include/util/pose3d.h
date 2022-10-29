#ifndef _POSE3D_H_
#define _POSE3D_H_

#include <iostream>
#include <vector>

#include <Eigen/Dense>

typedef Eigen::Vector3f Pos3f;

typedef Eigen::Vector4f RotQuatf;

typedef Eigen::Vector3f so3f;
typedef Eigen::Matrix3f SO3f;

typedef Eigen::Matrix<float,6,1> se3f;
typedef Eigen::Matrix<float,4,4> SE3f;

#define EPSILON_FLOAT 1e-5f;


class Rot3D
{
private:
    SO3f R_; // SO3
    RotQuatf q_; // w x y z (normalized quaternion, ||q||_2 = 1)

public:
    Rot3D()
    {
        R_= SO3f::Identity();
        q_ << 1.0f, 0.0f, 0.0f, 0.0f;
    };

    // Copy constructor
    Rot3D(const Rot3D& rot3d);


    ~Rot3D() { };

// Get methods
public:
    const SO3f&     R() const { return R_; };
    const RotQuatf& q() const { return q_; };
    const SO3f&     inverse() { return R_.transpose(); };

// Set method
public:
    void setIdentity() 
    { 
        R_ = SO3f::Identity(); 
        q_ << 1.0f, 0.0f, 0.0f, 0.0f;
    };

// Operator overloading
public:
    Rot3D& operator=(const Rot3D& rot3d)
    {
        R_ << rot3d.R();
        q_ << rot3d.q();
        
        // Normalize and project SO(3) space
        q_.normalize();



        return *this;
    };

    // 곱셈 연산자.
    Rot3D operator * (Rot3D& rot)
    {
        Rot3D rot_new;
        
        return rot_new;
    };

    friend std::ostream& operator << (std::ostream& os, const Rot3D& rot)
    {
        os << rot.R_;
        return os;
    };

};




class Pose3D
{
public:

private:
    Rot3D rot_; // SO3
    Pos3f t_; // 3d translation vector
    
    se3f xi_;
    SE3f T_;

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
    const SO3f& R() const     { return rot_.R(); };
    const Pos3f& t() const    { return t_; };
    const RotQuatf& q() const { return rot_.q(); };
    


private:


};

#endif