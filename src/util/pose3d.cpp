#include "util/pose3d.h"

Rotation3:: Rotation3() {
    R_= SO3::Identity();
    q_ << 1.0, 0.0, 0.0, 0.0;
    w_ << 0.0, 0.0, 0.0;
};

Rotation3::Rotation3(const SO3& R) {
    initByRotation(R);
};

Rotation3::Rotation3(const so3& w) {
    initByAxisAngle(w);
};

Rotation3::Rotation3(const Quaternion4& q){
    initByQuaternion(q);
};

// Copy constructor
Rotation3::Rotation3(const Rotation3& rot)
{
    q_ << rot.q();
    R_ << rot.R();        
    w_ << rot.w();
};

const SO3&         Rotation3::R() const { return R_; };
const Quaternion4& Rotation3::q() const { return q_; };
const so3&         Rotation3::w() const { return w_; };

Rotation3          Rotation3::inverse() const { 
    Quaternion4 qinv;
    qinv << q_(0), -q_(1), -q_(2), -q_(3);

    Rotation3 rot_inv;
    rot_inv.initByQuaternion(qinv);
    
    return rot_inv;
};

_mat_numeric Rotation3::determinant() const
{
    _mat_numeric val = 0.0;
    val += R_(0,0)*( R_(1,1)*R_(2,2) - R_(1,2)*R_(2,1));
    val -= R_(0,1)*( R_(1,0)*R_(2,2) - R_(1,2)*R_(2,0));
    val += R_(0,2)*( R_(1,0)*R_(2,1) - R_(1,1)*R_(2,0));
    return val;
};

void Rotation3::setIdentity()
{ 
    R_ = SO3::Identity(); 
    q_ << 1.0f, 0.0f, 0.0f, 0.0f;
    w_.setZero();
};

void Rotation3::initByRotation(const SO3& R)
{
    R_ << R;
    convertRotationToQuaternion();
    convertQuaternionToRotation();
    convertQuaternionToAxisAngle();
};

void Rotation3::initByQuaternion(const Quaternion4& q)
{
    q_ << q;
    q_.normalize();
    convertQuaternionToRotation();
    convertQuaternionToAxisAngle();
};

void Rotation3::initByAxisAngle(const so3& w)
{
    w_ << w;
    convertAxisAngleToQuaternion();
    convertQuaternionToRotation();
};

Rotation3& Rotation3::operator=(const Rotation3& Rotation3)
{
    initByQuaternion(Rotation3.q());
    return *this;
};

// 곱셈 연산자.
Rotation3 Rotation3::operator*(const Rotation3& rot) const
{
    Rotation3 rot_res;
    rot_res.initByQuaternion(q1_mult_q2(q_, rot.q()));
    return rot_res;
};

// 곱셈 대입 연산자.
Rotation3& Rotation3::operator*=(const Rotation3& rot)
{
    q_ = q1_mult_q2(q_,rot.q());
    q_.normalize(); // normalize
    convertQuaternionToRotation();
    return *this;
};

// 대입 연산자.
void Rotation3::operator<<(const Rotation3& rot){ // 대입 rot
    initByQuaternion(rot.q());
};

void Rotation3::operator<<(const SO3& R){ // 대입 R
    initByRotation(R);
};

void Rotation3::operator<<(const Quaternion4& q){ // 대입 q
    initByQuaternion(q);
};

void Rotation3::operator<<(const so3& w){
    initByAxisAngle(w);
};

Pose3 Rotation3::operator,(const Position3& t)
{
    return Pose3(*this, t);
};

std::ostream& operator << (std::ostream& os, const Rotation3& rot){
    os << rot.R_;
    return os;
};

void Rotation3::convertRotationToQuaternion()
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

void Rotation3::convertQuaternionToRotation()
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

void Rotation3::convertAxisAngleToQuaternion()
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

void Rotation3::convertQuaternionToAxisAngle()
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

Quaternion4 Rotation3::q1_mult_q2(const Quaternion4& q1, const Quaternion4& q2) const
{
    Quaternion4 q_res;
    q_res <<
        q1(0)*q2(0)-q1(1)*q2(1)-q1(2)*q2(2)-q1(3)*q2(3),
        q1(0)*q2(1)+q1(1)*q2(0)+q1(2)*q2(3)-q1(3)*q2(2),
        q1(0)*q2(2)-q1(1)*q2(3)+q1(2)*q2(0)+q1(3)*q2(1),
        q1(0)*q2(3)+q1(1)*q2(2)-q1(2)*q2(1)+q1(3)*q2(0);
    return q_res;
};





/*
=============================================
pose3
=============================================
*/
Pose3::Pose3() 
: rot_() {
    t_.setZero();
    T_.setZero();
};

Pose3::Pose3(const Rotation3& rot, const Position3& t) {
    initByQuaternionAndTranslation(rot.q(), t);
};

Pose3::Pose3(const SO3& R, const Position3& t) {
    initByRotationAndTranslation(R, t);
};

Pose3::Pose3(const Quaternion4& q, const Position3& t) {
    initByQuaternionAndTranslation(q,t);
};

Pose3::Pose3(const so3& w, const Position3& t){
    initByAxisAngleAndTranslation(w,t);
};

Pose3::Pose3(const Pose3& pose)
{
    rot_ = pose.rotation();
    t_  << pose.t();
    T_  << pose.T();  
};


const Rotation3& Pose3::rotation() const { return rot_; };
const Quaternion4&      Pose3::q() const { return rot_.q(); };

const SO3&       Pose3::R() const { return rot_.R(); };
const Position3& Pose3::t() const { return t_; };
const SE3&       Pose3::T() const { return T_; };

Pose3 Pose3::inverse() const {
    Rotation3 rot_inv = rot_.inverse();
    Pose3 pose_inv( rot_inv, -rot_inv.R()*t_);
    
    return pose_inv;
};

void Pose3::setIdentity()
{
    rot_.setIdentity();
    t_ << 0,0,0;
    T_ << rot_.R(), t_, 0.0, 0.0, 0.0, 1.0;
};

void Pose3::initByRotationAndTranslation(const SO3& R, const Position3& t)
{
    rot_ << R;
    t_ << t;
    
    T_ << rot_.R(), t_, 0,0,0,1;
};

void Pose3::initByQuaternionAndTranslation(const Quaternion4& q, const Position3& t)
{
    rot_ << q;
    t_ << t;
    
    T_ << rot_.R(), t_, 0,0,0,1;
};

void Pose3::initByAxisAngleAndTranslation(const so3& w, const Position3& t)
{
    rot_ << w;
    t_ << t;
    
    T_ << rot_.R(), t_, 0,0,0,1;
};


Pose3& Pose3::operator=(const Pose3& pose)
{
    initByQuaternionAndTranslation(pose.rotation().q(), pose.t());
    return *this;
};

// 곱셈 연산자
Pose3 Pose3::operator*(const Pose3& pose) const
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
Pose3& Pose3::operator*=(const Pose3& pose)
{
    t_.noalias() = t_ + rot_.R()*pose.t();

    rot_ *= pose.rotation();

    this->initByQuaternionAndTranslation(rot_.q(), t_);

    return *this;
};


// 대입 연산자.
void Pose3::operator<<(const Pose3& pose){ // 대입 rot
    initByQuaternionAndTranslation(pose.q(), pose.t());
};


std::ostream& operator << (std::ostream& os, const Pose3& pose)
{
    os << pose.T();
    return os;
};
