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
class Pose3;

class Rotation3
{
private:
    SO3 R_; // SO3
    Quaternion4 q_; // w x y z (normalized quaternion, ||q||_2 = 1)
    so3 w_;

public:
    Rotation3()
    {
        R_= SO3::Identity();
        q_ << 1.0, 0.0, 0.0, 0.0;
        w_ << 0.0, 0.0, 0.0;
    };

    // Copy constructor
    Rotation3(const Rotation3& rot)
    {
        q_ << rot.q();
        R_ << rot.R();        
        w_ << rot.w();
    };

// Get methods
public:
    const SO3&         R() const { return R_; };
    const Quaternion4& q() const { return q_; };
    const so3&         w() const { return w_; };

    Rotation3          inverse() const { 
        Quaternion4 qinv;
        qinv << q_(0), -q_(1), -q_(2), -q_(3);

        Rotation3 rot_inv;
        rot_inv.initByQuaternion(qinv);
        
        return rot_inv; 
    };

    _mat_numeric determinant() const
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
        w_.setZero();
    };

    void initByRotation(const SO3& R)
    {
        R_ << R;
        convertRotationToQuaternion();
        convertQuaternionToRotation();
        convertQuaternionToAxisAngle();
    };

    void initByQuaternion(const Quaternion4& q)
    {
        q_ << q;
        q_.normalize();
        convertQuaternionToRotation();
        convertQuaternionToAxisAngle();
    };

    void initByAxisAngle(const so3& w)
    {
        w_ << w;
        convertAxisAngleToQuaternion();
        convertQuaternionToRotation();
    };

// Operator overloading
// 대입 연산 (깊은 복사)
// 곱셈 연산
// 곱셈 대입 연산 (자기 자신에게 곱셈하고 자기 자신을 리턴)
public:
    Rotation3& operator=(const Rotation3& Rotation3)
    {
        initByQuaternion(Rotation3.q());
        q_.normalize(); // normalize
        
        convertQuaternionToRotation();        
        return *this;
    };

    // 곱셈 연산자.
    Rotation3 operator*(const Rotation3& rot) const
    {
        Rotation3 rot_res;
        rot_res.initByQuaternion(q1_mult_q2(q_, rot.q()));
        return rot_res;
    };

    // 곱셈 대입 연산자.
    Rotation3& operator*=(const Rotation3& rot)
    {
        q_ = q1_mult_q2(q_,rot.q());
        q_.normalize(); // normalize
        convertQuaternionToRotation();
        return *this;
    };

    // 대입 연산자.
    void operator<<(const Rotation3& rot) // 대입 rot
    {
        initByQuaternion(rot.q());
    };
    
    void operator<<(const SO3& R) // 대입 R
    {
        initByRotation(R);
    };

    void operator<<(const Quaternion4& q) // 대입 q
    {
        initByQuaternion(q);
    };

    void operator<<(const so3& w)
    {
        initByAxisAngle(w);
    };

    friend std::ostream& operator << (std::ostream& os, const Rotation3& rot)
    {
        os << rot.R_;
        return os;
    };

private:
    void convertRotationToQuaternion()
    {
        const _mat_numeric& m00 = R_(0,0);
        const _mat_numeric& m11 = R_(1,1);
        const _mat_numeric& m22 = R_(2,2);

        const _mat_numeric& m21 = R_(2,1);
        const _mat_numeric& m12 = R_(1,2);
        const _mat_numeric& m02 = R_(0,2);
        const _mat_numeric& m20 = R_(2,0);
        const _mat_numeric& m10 = R_(1,0);
        const _mat_numeric& m01 = R_(0,1);

        /*
             To resolve the signs, find the largest of q0, q1, q2, q3 and 
            assume its sign is positive. 
             Then compute the remaining components as shown in the table below.
             Taking the largest magnitude avoids division by small numbers, 
            which would reduce numerical accuracy. 
            https://danceswithcode.net/engineeringnotes/quaternions/quaternions.html
            https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/

            -   Avoid division by zero - We need to be sure that S is never zero even with possible floating point errors or de-orthogonalised matrix input.
            -   Avoid square root of a negative number. - We need to be sure that the tr value chosen is never negative even with possible floating point errors or de-orthogonalised matrix input.
            -   Accuracy of dividing by (and square rooting) a very small number. 
                with floating point numbers, dividing small numbers by small numbers should be reasonably accurate but at the extreme it would loose accuracy.
            -   Resilient to a de-orthogonalised matrix
        */

        _mat_numeric tr = m00 + m11 + m22;
        
        _mat_numeric S, Sinv;
        _mat_numeric w, x, y, z;
        if ( tr > 0 ) // tr > 0
        {  
            S = std::sqrt(tr + 1.0) * 2.0; // S=4*qw 
            Sinv = 1.0 / S;
            w = 0.25 * S;
            x = (m21 - m12) * Sinv;
            y = (m02 - m20) * Sinv; 
            z = (m10 - m01) * Sinv; 
        }
        else if ( (m00 > m11) & (m00 > m22) ) 
        {
            S = std::sqrt(1.0 + m00 - m11 - m22) * 2.0; // S=4*qx 
            Sinv = 1.0 / S;
            w = (m21 - m12) * Sinv;
            x = 0.25 * S;
            y = (m01 + m10) * Sinv; 
            z = (m02 + m20) * Sinv; 
        }
        else if (m11 > m22) 
        { 
            S = std::sqrt(1.0 + m11 - m00 - m22) * 2.0; // S=4*qy
            Sinv = 1.0 / S;
            w = (m02 - m20) * Sinv;
            x = (m01 + m10) * Sinv; 
            y = 0.25 * S;
            z = (m12 + m21) * Sinv; 
        }
        else 
        {
            S = std::sqrt(1.0 + m22 - m00 - m11) * 2.0; // S=4*qz
            Sinv = 1.0 / S;
            w = (m10 - m01) * Sinv;
            x = (m02 + m20) * Sinv;
            y = (m12 + m21) * Sinv;
            z = 0.25 * S;
        }

        q_(0) = w;
        q_(1) = x;
        q_(2) = y;
        q_(3) = z;

        q_.normalize();
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

    void convertAxisAngleToQuaternion()
    {
        _mat_numeric theta = w_.norm();

        theta = std::sqrt(theta);
        if(theta < FLT_EPSILON)
        {
            q_ << 1,0,0,0;
        }
        else
        {
            _mat_numeric invthsinth05 = sin(theta*0.5)/theta;
            q_ << cos(theta*0.5), w_(0)*invthsinth05, w_(1)*invthsinth05, w_(2)*invthsinth05;
            q_ /= q_.norm();
        }
    };

    void convertQuaternionToAxisAngle()
    {
        const _mat_numeric& w = q_(0);
        const _mat_numeric& x = q_(1);
        const _mat_numeric& y = q_(2);
        const _mat_numeric& z = q_(3);

        _mat_numeric theta;
        if( abs(w) < 0.5 )
        {
            theta = acos(w);
            _mat_numeric sinth = sin(theta);
            _mat_numeric th_invsinth = theta/sinth;

            w_ << x*th_invsinth, y*th_invsinth, z*th_invsinth;            
        }
        else
        {
            _mat_numeric sinth = sqrt(x*x + y*y + z*z);
            theta = asin(sinth);
            _mat_numeric th_invsinth = theta/sinth;

            w_ << x*th_invsinth, y*th_invsinth, z*th_invsinth;            
        }
    };

    Quaternion4 q1_mult_q2(const Quaternion4& q1, const Quaternion4& q2) const
    {
        Quaternion4 q_res;
        q_res <<
            q1(0)*q2(0)-q1(1)*q2(1)-q1(2)*q2(2)-q1(3)*q2(3),
            q1(0)*q2(1)+q1(1)*q2(0)+q1(2)*q2(3)-q1(3)*q2(2),
            q1(0)*q2(2)-q1(1)*q2(3)+q1(2)*q2(0)+q1(3)*q2(1),
            q1(0)*q2(3)+q1(1)*q2(2)-q1(2)*q2(1)+q1(3)*q2(0);
        return q_res;
    };
};


class Pose3
{
private:
    Rotation3 rot_; // SO3

    Position3 t_; // 3d translation vector
    
    SE3 T_;

public:
    Pose3() : rot_()
    {
        t_.setZero();
        T_.setZero();
    };

    Pose3(const Pose3& pose)
    {
        rot_ = pose.rotation();
        t_  << pose.t();
        T_  << pose.T();  
    };
    

// Get method (const)
public:
    const Rotation3& rotation() const { return rot_; };
    const SO3& R() const              { return rot_.R(); };
    const Quaternion4& q() const      { return rot_.q(); };

    const Position3& t() const { return t_; };
    const SE3& T() const       { return T_; };
    
    Pose3 inverse() const {
        Pose3 pose_inv;
        // pose_inv.init

        return pose_inv;
    };

// Set methods
public:
    void setIdentity()
    {
        rot_.setIdentity();
        t_ << 0,0,0;
        T_ << rot_.R(), t_, 0.0, 0.0, 0.0, 1.0;
    };

    void initByRotationAndTranslation(const SO3& R, const Position3& t)
    {
        rot_ << R;
        t_ << t;
        
        T_ << rot_.R(), t_, 0,0,0,1;
        // xi_ << se3Log();
    };

    void initByQuaternionAndTranslation(const Quaternion4& q, const Position3& t)
    {
        rot_ << q;
        t_ << t;
        
        T_ << rot_.R(), t_, 0,0,0,1;
        // xi_ << se3Log();
    };

public:
    Pose3& operator=(const Pose3& pose)
    {
        initByQuaternionAndTranslation(pose.rotation().q(), pose.t());
        return *this;
    };

    // 곱셈 연산자
    Pose3 operator*(const Pose3& pose) const
    {
        Rotation3 rot_res;
        rot_res = rot_ * pose.rotation();

        Position3 t_res;
        t_res << t_ + rot_.R()*pose.t();

        Pose3 pose_res;
        pose_res.initByQuaternionAndTranslation(rot_res.q(), t_res);

        return pose_res;  
    };

    // 곱셈 대입 연산자
    Pose3& operator*=(const Pose3& pose)
    {
        t_.noalias() = t_ + rot_.R()*pose.t();

        rot_ *= pose.rotation();

        this->initByQuaternionAndTranslation(rot_.q(), t_);

        return *this;
    };






    friend std::ostream& operator << (std::ostream& os, const Pose3& pose)
    {
        os << pose.T();
        return os;
    };

private:


};

#endif