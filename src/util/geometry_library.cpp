#include "util/geometry_library.h"
using namespace Eigen;
namespace geometry {

    Matrix3d skewMat(const Vector3d& v){
        Matrix3d res_mat;
        res_mat << 0,-v(2),v(1),
                   v(2),0,-v(0),
                  -v(1),v(0),0;
        return res_mat;
    };
    Matrix3f skewMat_f(const Vector3f& v){
        Matrix3f res_mat;
        res_mat << 0,-v(2),v(1),
                   v(2),0,-v(0),
                  -v(1),v(0),0;
        return res_mat;
    };

    Matrix4d q_right_mult(const Vector4d& q){
        Matrix4d omega_r;
        omega_r << q(0),-q(1),-q(2),-q(3),
                   q(1), q(0), q(3),-q(2),
                   q(2),-q(3), q(0), q(1),
                   q(3), q(2),-q(1), q(0);
        return omega_r;
    };
    Matrix4f q_right_mult_f(const Vector4f& q){
        Matrix4f omega_r;
        omega_r << q(0),-q(1),-q(2),-q(3),
                   q(1), q(0), q(3),-q(2),
                   q(2),-q(3), q(0), q(1),
                   q(3), q(2),-q(1), q(0);
        return omega_r;
    };

    Matrix4d q_left_mult(const Vector4d& q){
        Matrix4d omega_l;
        omega_l << q(0),-q(1),-q(2),-q(3),
                   q(1), q(0),-q(3), q(2),
                   q(2), q(3), q(0),-q(1),
                   q(3),-q(2), q(1), q(0);
        return omega_l;
    };
    Matrix4f q_left_mult_f(const Vector4f& q){
        Matrix4f omega_l;
        omega_l << q(0),-q(1),-q(2),-q(3),
                   q(1), q(0),-q(3), q(2),
                   q(2), q(3), q(0),-q(1),
                   q(3),-q(2), q(1), q(0);
        return omega_l;
    };

    Vector4d q_conj(const Vector4d& q){
        Vector4d q_c;
        q_c << q(0), -q(1),-q(2),-q(3);
        return q_c;
    };
    Vector4f q_conj_f(const Vector4f& q){
        Vector4f q_c;
        q_c << q(0), -q(1),-q(2),-q(3);
        return q_c;
    };

    Vector4d q1_mult_q2(const Vector4d& q1, const Vector4d& q2){
        Vector4d q;
        q << q1(0)*q2(0)-q1(1)*q2(1)-q1(2)*q2(2)-q1(3)*q2(3),
             q1(0)*q2(1)+q1(1)*q2(0)+q1(2)*q2(3)-q1(3)*q2(2),
             q1(0)*q2(2)-q1(1)*q2(3)+q1(2)*q2(0)+q1(3)*q2(1),
             q1(0)*q2(3)+q1(1)*q2(2)-q1(2)*q2(1)+q1(3)*q2(0);
        return q;
    };
    Vector4f q1_mult_q2_f(const Vector4f& q1, const Vector4f& q2){
        Vector4f q;
        q << q1(0)*q2(0)-q1(1)*q2(1)-q1(2)*q2(2)-q1(3)*q2(3),
             q1(0)*q2(1)+q1(1)*q2(0)+q1(2)*q2(3)-q1(3)*q2(2),
             q1(0)*q2(2)-q1(1)*q2(3)+q1(2)*q2(0)+q1(3)*q2(1),
             q1(0)*q2(3)+q1(1)*q2(2)-q1(2)*q2(1)+q1(3)*q2(0);
        return q;
    };

    Matrix3d q2r(const Vector4d& q){
        Matrix3d R;
        double qw = q(0);
        double qx = q(1);
        double qy = q(2);
        double qz = q(3);

        double qw2 = qw*qw;
        double qx2 = qx*qx;
        double qy2 = qy*qy;
        double qz2 = qz*qz;

        double qxqy = qx*qy;
        double qwqz = qw*qz;
        double qxqz = qx*qz;
        double qwqy = qw*qy;
        double qwqx = qw*qx;
        double qyqz = qy*qz;

        R <<  qw2+qx2-qy2-qz2, 2.0*(qxqy-qwqz), 2.0*(qxqz+qwqy),
              2.0*(qxqy+qwqz), qw2-qx2+qy2-qz2, 2.0*(qyqz-qwqx),
              2.0*(qxqz-qwqy), 2.0*(qyqz+qwqx), qw2-qx2-qy2+qz2;    

        return R;
    };
    Matrix3f q2r_f(const Vector4f& q){
        Matrix3f R;
        float qw = q(0);
        float qx = q(1);
        float qy = q(2);
        float qz = q(3);

        float qw2 = qw*qw;
        float qx2 = qx*qx;
        float qy2 = qy*qy;
        float qz2 = qz*qz;

        float qxqy = qx*qy;
        float qwqz = qw*qz;
        float qxqz = qx*qz;
        float qwqy = qw*qy;
        float qwqx = qw*qx;
        float qyqz = qy*qz;

        R <<  qw2+qx2-qy2-qz2, 2.0*(qxqy-qwqz), 2.0*(qxqz+qwqy),
              2.0*(qxqy+qwqz), qw2-qx2+qy2-qz2, 2.0*(qyqz-qwqx),
              2.0*(qxqz-qwqy), 2.0*(qyqz+qwqx), qw2-qx2-qy2+qz2;    

        return R;
    };


    Vector4d rotvec2q(const Vector3d& w){
        Vector4d q_res;
        double th = w(0)*w(0) + w(1)*w(1) + w(2)*w(2);
        th = std::sqrt(th);
        if(th < 1e-7){
            q_res << 1.0, 0.0, 0.0, 0.0;
        }
        else{
            double invthsinth05 = sin(th*0.5)/th;
            q_res << cos(th*0.5),w(0)*invthsinth05, w(1)*invthsinth05, w(2)*invthsinth05;
            q_res /= q_res.norm();
        }
        return q_res;
    };
    Vector4f rotvec2q_f(const Vector3f& w){
        Vector4f q_res;
        float th = w(0)*w(0) + w(1)*w(1) + w(2)*w(2);
        th = std::sqrt(th);
        if(th < 1e-7){
            q_res << 1.0, 0.0, 0.0, 0.0;
        }
        else{
            float invthsinth05 = sin(th*0.5)/th;
            q_res << cos(th*0.5),w(0)*invthsinth05, w(1)*invthsinth05, w(2)*invthsinth05;
            q_res /= q_res.norm();
        }
        return q_res;
    };

    Matrix3d a2r(double r, double p, double y){
        Matrix3d Rx;
        Matrix3d Ry;
        Matrix3d Rz;

        Rx << 1,0,0,0,cos(r),-sin(r),0,sin(r),cos(r);
        Ry << cos(p),0,sin(p),0,1,0,-sin(p),0,cos(p);
        Rz << cos(y),-sin(y),0,sin(y),cos(y),0,0,0,1;

        return (Rz*Ry*Rx);
    }; 
    Matrix3f a2r_f(float r, float p, float y){
        Matrix3f Rx;
        Matrix3f Ry;
        Matrix3f Rz;

        Rx << 1,0,0,0,cos(r),-sin(r),0,sin(r),cos(r);
        Ry << cos(p),0,sin(p),0,1,0,-sin(p),0,cos(p);
        Rz << cos(y),-sin(y),0,sin(y),cos(y),0,0,0,1;

        return (Rz*Ry*Rx);
    };

    Vector4d r2q(const Matrix3d& R){
        Vector4d q;
        double qw,qx,qy,qz;

        double m00 = R(0,0);
        double m11 = R(1,1);
        double m22 = R(2,2);

        double m21 = R(2,1);
        double m12 = R(1,2);
        double m02 = R(0,2);
        double m20 = R(2,0);
        double m10 = R(1,0);
        double m01 = R(0,1);

        double tr = R(0,0) + R(1,1) + R(2,2);

        if (tr > 0) { 
            double S = sqrt(tr+1.0) * 2; // S=4*qw 
            qw = 0.25 * S;
            qx = (m21 - m12) / S;
            qy = (m02 - m20) / S; 
            qz = (m10 - m01) / S; 
        } else if ((m00 > m11)&(m00 > m22)) { 
            double S = sqrt(1.0 + m00 - m11 - m22) * 2; // S=4*qx 
            qw = (m21 - m12) / S;
            qx = 0.25 * S;
            qy = (m01 + m10) / S; 
            qz = (m02 + m20) / S; 
        } else if (m11 > m22) { 
            double S = sqrt(1.0 + m11 - m00 - m22) * 2; // S=4*qy
            qw = (m02 - m20) / S;
            qx = (m01 + m10) / S; 
            qy = 0.25 * S;
            qz = (m12 + m21) / S; 
        } else { 
            double S = sqrt(1.0 + m22 - m00 - m11) * 2; // S=4*qz
            qw = (m10 - m01) / S;
            qx = (m02 + m20) / S;
            qy = (m12 + m21) / S;
            qz = 0.25 * S;
        }

        q(0) = qw;
        q(1) = qx;
        q(2) = qy;
        q(3) = qz;
        return q;
    };


    Vector4f r2q_f(const Matrix3f& R){
        Vector4f q;
        float qw,qx,qy,qz;

        float m00 = R(0,0);
        float m11 = R(1,1);
        float m22 = R(2,2);

        float m21 = R(2,1);
        float m12 = R(1,2);
        float m02 = R(0,2);
        float m20 = R(2,0);
        float m10 = R(1,0);
        float m01 = R(0,1);

        float tr = R(0,0) + R(1,1) + R(2,2);

        if (tr > 0) { 
            float S = sqrt(tr+1.0) * 2; // S=4*qw 
            qw = 0.25 * S;
            qx = (m21 - m12) / S;
            qy = (m02 - m20) / S; 
            qz = (m10 - m01) / S; 
        } else if ((m00 > m11)&(m00 > m22)) { 
            float S = sqrt(1.0 + m00 - m11 - m22) * 2; // S=4*qx 
            qw = (m21 - m12) / S;
            qx = 0.25 * S;
            qy = (m01 + m10) / S; 
            qz = (m02 + m20) / S; 
        } else if (m11 > m22) { 
            float S = sqrt(1.0 + m11 - m00 - m22) * 2; // S=4*qy
            qw = (m02 - m20) / S;
            qx = (m01 + m10) / S; 
            qy = 0.25 * S;
            qz = (m12 + m21) / S; 
        } else { 
            float S = sqrt(1.0 + m22 - m00 - m11) * 2; // S=4*qz
            qw = (m10 - m01) / S;
            qx = (m02 + m20) / S;
            qy = (m12 + m21) / S;
            qz = 0.25 * S;
        }

        q(0) = qw;
        q(1) = qx;
        q(2) = qy;
        q(3) = qz;
        return q;
    };

    void se3Exp_f(const Eigen::Matrix<float,6,1>& xi, Eigen::Matrix4f& T){
        // initialize variables
        float theta = 0.0;
        Eigen::Vector3f v, w;
        Eigen::Matrix3f wx, R, V;
        Eigen::Vector3f t;

        v(0) = xi(0);
        v(1) = xi(1);
        v(2) = xi(2);

        w(0) = xi(3);
        w(1) = xi(4);
        w(2) = xi(5);

        theta = std::sqrt(w.transpose() * w);
        wx << 0, -w(2), w(1),
            w(2), 0, -w(0),
            -w(1), w(0), 0;

        if (theta < 1e-7) {
            R = Eigen::Matrix3f::Identity() + wx + 0.5 * wx * wx;
            V = Eigen::Matrix3f::Identity() + 0.5 * wx + wx * wx *0.33333333333333333333333333f;
        }
        else {
            R = Eigen::Matrix3f::Identity() + (sin(theta) / theta) * wx + ((1 - cos(theta)) / (theta*theta)) * (wx*wx);
            V = Eigen::Matrix3f::Identity() + ((1 - cos(theta)) / (theta*theta)) * wx + ((theta - sin(theta)) / (theta*theta*theta)) * (wx*wx);
        }
        t = V * v;

        // assign rigid body transformation matrix (in SE(3))
        T = Eigen::Matrix4f::Zero();
        T(0, 0) = R(0, 0);
        T(0, 1) = R(0, 1);
        T(0, 2) = R(0, 2);

        T(1, 0) = R(1, 0);
        T(1, 1) = R(1, 1);
        T(1, 2) = R(1, 2);

        T(2, 0) = R(2, 0);
        T(2, 1) = R(2, 1);
        T(2, 2) = R(2, 2);

        T(0, 3) = t(0);
        T(1, 3) = t(1);
        T(2, 3) = t(2);

        T(3, 3) = 1.0f;

        // for debug
        // std::cout << R << std::endl;
        // std::cout << t << std::endl;
        //usleep(10000000);
    };

    void SE3Log_f(const Eigen::Matrix<float,4,4>& T, Eigen::Matrix<float,6,1>& xi){
        float theta = 0.0f;
        float A, B;
        Eigen::Matrix3f wx, R;
        Eigen::MatrixXf lnR, Vin;
        Eigen::Vector3f t, w, v;

        R = T.block<3, 3>(0, 0);
        t = T.block<3, 1>(0, 3);


        theta = acosf((R.trace() - 1.0f)*0.5f);

        if (theta < 1e-9) {
            Vin = Eigen::MatrixXf::Identity(3, 3);
            w << 0, 0, 0;
        }
        else {
            lnR = (theta / (2 * sinf(theta)))*(R - R.transpose());
            w << -lnR(1, 2), lnR(0, 2), -lnR(0, 1);
            wx << 0, -w(2), w(1),
                w(2), 0, -w(0),
                -w(1), w(0), 0;
            A = sinf(theta) / theta;
            B = (1 - cosf(theta)) / (theta*theta);
            Vin = Eigen::MatrixXf::Identity(3, 3) - 0.5f*wx + (1.0f / (theta*theta))*(1.0f - A / (2.0f * B))*(wx*wx);
        }

        v = Vin*t;
        xi.setZero();
        xi(0) = v(0);
        xi(1) = v(1);
        xi(2) = v(2);
        xi(3) = w(0);
        xi(4) = w(1);
        xi(5) = w(2);
        // for debug
        // std::cout << xi << std::endl;
    };

    void addFrontse3_f(Eigen::Matrix<float,6,1>& xi, const Eigen::Matrix<float,6,1>& dxi){
        Eigen::Matrix<float,4,4> Tjw, dT;
        se3Exp_f(xi,  Tjw);
        se3Exp_f(dxi, dT);
        Tjw.noalias() = dT*Tjw;
        SE3Log_f(Tjw, xi);
    };

};