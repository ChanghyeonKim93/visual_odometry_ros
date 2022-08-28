#include "core/mapping.h"

namespace Mapping{
    Eigen::MatrixXf m_matrix_template_;
    void triangulateDLT(const PixelVec& pts0, const PixelVec& pts1, 
                        const Eigen::Matrix3f& R10, const Eigen::Vector3f& t10, const std::shared_ptr<Camera>& cam, 
                        PointVec& X0, PointVec& X1)
    {
        if(pts0.size() != pts1.size() )
            throw std::runtime_error("pts0.size() != pts1.size()");

        int n_pts = pts0.size(); 

        float fx = cam->fx(); float fy = cam->fy();
        float cx = cam->cx(); float cy = cam->cy();

        Eigen::Matrix<float,3,4> P00;
        Eigen::Matrix<float,3,4> P10;
        
        P00 << cam->K(),Eigen::Vector3f::Zero();
        P10 << cam->K()*R10, cam->K()*t10;

        Eigen::Matrix4f M; M.setZero();
        // Constant elements
        M(0,0) = -fx; M(0,1) =   0; M(0,3) = 0; 
        M(1,0) =   0; M(1,1) = -fy; M(1,3) = 0; 

        X0.resize(n_pts);
        X1.resize(n_pts);
        for(int i = 0; i < n_pts; ++i){
            const float& u0 = pts0[i].x; const float& v0 = pts0[i].y;
            const float& u1 = pts1[i].x; const float& v1 = pts1[i].y;

            // M(0,0) = -fx; M(0,1) =   0; M(0,2) = u0-cx; M(0,3) = 0; 
            // M(1,0) =   0; M(1,1) = -fy; M(1,2) = v0-cy; M(1,3) = 0; 
            M(0,2) = u0 - cx; 
            M(1,2) = v0 - cy; 
            M.block<1,4>(2,0) = u1*P10.block<1,4>(2,0) - P10.block<1,4>(0,0);
            M.block<1,4>(3,0) = v1*P10.block<1,4>(2,0) - P10.block<1,4>(1,0);
            
            // Solve SVD
            Eigen::MatrixXf V;
            Eigen::JacobiSVD<Eigen::MatrixXf> svd(M, Eigen::ComputeFullV);
            V = svd.matrixV();     

            // Get 3D point
            X0[i] = V.block<3,1>(0,3)/V(3,3);       
            X1[i] = R10*X0[i] + t10;
        }
   };

    void triangulateDLT(const Pixel& pt0, const Pixel& pt1, 
                        const Eigen::Matrix3f& R10, const Eigen::Vector3f& t10, const std::shared_ptr<Camera>& cam, 
                        Point& X0, Point& X1)
    {
        const float& fx = cam->fx(); const float& fy = cam->fy();
        const float& cx = cam->cx(); const float& cy = cam->cy();

        Eigen::Matrix<float,3,4> P00;
        Eigen::Matrix<float,3,4> P10;
        
        P00 << cam->K(),Eigen::Vector3f::Zero();
        P10 << cam->K()*R10, cam->K()*t10;

        Eigen::Matrix4f M;  M.setZero();
        // Constant elements
        M(0,0) = -fx; M(0,1) =   0; M(0,3) = 0; 
        M(1,0) =   0; M(1,1) = -fy; M(1,3) = 0; 

        const float& u0 = pt0.x; const float& v0 = pt0.y;
        const float& u1 = pt1.x; const float& v1 = pt1.y;

        // M(0,0) = -fx; M(0,1) =   0; M(0,2) = u0-cx; M(0,3) = 0; 
        // M(1,0) =   0; M(1,1) = -fy; M(1,2) = v0-cy; M(1,3) = 0; 
        M(0,2) = u0 - cx; 
        M(1,2) = v0 - cy; 
        M.block<1,4>(2,0) = u1*P10.block<1,4>(2,0) - P10.block<1,4>(0,0);
        M.block<1,4>(3,0) = v1*P10.block<1,4>(2,0) - P10.block<1,4>(1,0);
        
        // Solve SVD
        Eigen::MatrixXf V;
        Eigen::JacobiSVD<Eigen::MatrixXf> svd(M, Eigen::ComputeFullV);
        V = svd.matrixV();     

        // Get 3D point
        X0 = V.block<3,1>(0,3)/V(3,3);       
        X1 = R10*X0 + t10;
   };

    Eigen::Matrix3f skew(const Eigen::Vector3f& vec){
        Eigen::Matrix3f mat;
        mat <<    0.0, -vec(2),  vec(1), 
            vec(2),     0.0, -vec(0),
            -vec(1),  vec(0),     0.0;
        return mat;
    };
};