#ifndef _BUNDLE_ADJUSTMENT_SOLVER_H_
#define _BUNDLE_ADJUSTMENT_SOLVER_H_

#include <iostream>
#include <vector>

#include <Eigen/Dense>
#include "util/geometry_library.h"
#include "util/timer.h"
#include "core/camera.h"
#include "core/type_defines.h"

typedef std::vector<Mat66>              BlockDiagonalMat66; 
typedef std::vector<Mat33>              BlockDiagonalMat33; 
typedef std::vector<std::vector<Mat63>> BlockFullMat63; 
typedef std::vector<std::vector<Mat36>> BlockFullMat36;
typedef std::vector<std::vector<Mat66>> BlockFullMat66;
typedef std::vector<Vec6>               BlockVec6;
typedef std::vector<Vec3>               BlockVec3;

// Dedicated solver class to solve Feature-based Bundle adjustment problem.
class BundleAdjustmentSolver{
private:
    std::shared_ptr<Camera> cam_;

private:
    /*
        <Problem we want to solve>
            H*delta_theta = J.transpose()*r;

        where
        - Hessian  
                H = [A_, B_;
                     Bt_, C_];

        - Jacobian multiplied by residual vector 
                J.transpose()*r = [a;b];

        - Update parameters
                delta_theta = [x;y];
    */
    BlockDiagonalMat66 A_; // N_opt (6x6) block diagonal part for poses
    BlockFullMat63     B_; // N_opt x M (6x3) block part (side)
    BlockFullMat36     Bt_; // M x N_opt (3x6) block part (side, transposed)
    BlockDiagonalMat33 C_; // M (3x3) block diagonal part for landmarks' 3D points
    
    BlockVec6 a_; // N_opt x 1 (6x1) 
    BlockVec3 b_; // M x 1 (3x1)

    // Variables to solve Schur Complement
    BlockDiagonalMat33 Cinv_; // block diagonal part for landmarks' 3D points (inverse)
    BlockFullMat63     BCinv_; // N_opt X M  (6x3)
    BlockFullMat66     BCinvBt_; // N_opt x N_opt (6x6)
    BlockVec6          BCinv_b_; // N_opt x 1 (6x1)
    BlockVec3          Bt_x_; // M x 1 (3x1)

    BlockVec6 x_; // N_opt blocks (6x1)
    BlockVec3 y_; // M blocks (3x1)

    BlockVec6 params_poses_; // parameter vector for poses
    BlockVec3 params_points_; // parameter vector for points

    // Variables
    LandmarkBAVec lms_ba_; // landmarks to be optimized
    std::map<FramePtr,PoseSE3> Tjw_map_; // map containing Tjw_map_
    std::map<FramePtr,int> kfmap_optimize_; // map containing optimizable keyframes and their indexes
    std::vector<FramePtr> kfvec_optimize_;

    // Problem size
    int N_;     // # of total poses including fixed poses
    int N_opt_; // # of poses to be optimized
    int M_;     // # of landmarks to be optimized
    int n_obs_; // # of observations

    // optimization parameters
    float THRES_HUBER_;

public:
    // Constructor for BundleAdjustmentSolver ( se3 (6-DoF), 3D points (3-DoF) )
    // Sparse solver with 6x6 block diagonals, 3x3 block diagonals, 6x3 and 3x6 blocks.
    BundleAdjustmentSolver() {
        cam_ = nullptr;

        A_.reserve(500); // reserve expected # of optimizable poses (N_opt)
        B_.reserve(500); // 
        Bt_.reserve(1000000);
        C_.reserve(1000000); // reserve expected # of optimizable landmarks (M)
    };

    void setHuberThreshold(float thres_huber){
        THRES_HUBER_ = thres_huber;
    };

    void setCamera(const std::shared_ptr<Camera>& cam){
        cam_ = cam;
    };

    // N    : # of total poses (including fixed poses)
    // N_opt: # of optimizable poses
    // M    : # of optimizable landmarks
    // n_obs: # of observations
    void setProblemSize(int N, int N_opt, int M, int n_obs){ 
        N_     = N; 
        N_opt_ = N_opt;
        M_     = M;

        A_.resize(N_opt_); 
        B_.resize(N_opt_); for(int j = 0; j < N_opt_; ++j) B_[j].resize(M_, Mat63::Zero());   // 6x3, N_opt X M blocks
        Bt_.resize(M_); for(int i = 0; i < M_; ++i) Bt_[i].resize(N_opt_, Mat36::Zero());   // 3x6, N_opt X M blocks

        C_.resize(M_);   

        a_.resize(N_opt_); // 6*N_opt x 1 
        b_.resize(M_); // 3*M x 1

        Cinv_.resize(M_); // 3x3, M diagonal blocks 
        BCinv_.resize(N_opt_); for(int j = 0; j < N_opt_; ++j) BCinv_[j].resize(M_, Mat63::Zero());   // 6x3, N_opt X M blocks
        BCinvBt_.resize(N_opt_); for(int j = 0; j < N_opt_; ++j) BCinvBt_[j].resize(N_opt_, Mat66::Zero());   // 6x6, N_opt X N_opt blocks
        BCinv_b_.resize(N_opt_); // 6x1, N_opt x 1 blocks
        Bt_x_.resize(M_);        // 3x1, M x 1 blocks

        x_.resize(N_opt_); // 6x1, N_opt blocks
        y_.resize(M_);     // 3x1, M blocks

        params_poses_.resize(N_opt_);
        params_points_.resize(M_);
    };

    void setInitialValues(
        const std::map<FramePtr,PoseSE3>& Tjw_map,
        const LandmarkBAVec& lms_ba,
        const std::map<FramePtr,int>& kfmap_optimize)
    {
        Tjw_map_ = Tjw_map;
        kfmap_optimize_ = kfmap_optimize;
        kfvec_optimize_.resize(kfmap_optimize_.size());
        for(auto kf : kfmap_optimize_){
            kfvec_optimize_[kf.second] = kf.first;
        }


        lms_ba_.resize(lms_ba.size());
        for(int i = 0; i < lms_ba.size(); ++i) lms_ba_[i] = lms_ba[i];

        if(M_ != lms_ba.size()) throw std::runtime_error("In BundleAdjustmentSolver, 'M_ != lms_ba.size()'.");
        if(N_ != Tjw_map_.size()) throw std::runtime_error("In BundleAdjustmentSolver, 'N_ != Tjw_map_.size()'.");

    };

    // Solve
    void solveInFiniteIterations(int MAX_ITER){
        timer::tic();        

        // Intrinsic of lower camera
        Mat33 K = cam_->K(); Mat33 Kinv = cam_->Kinv();
        float fx = cam_->fx(); float fy = cam_->fy();
        float cx = cam_->cx(); float cy = cam_->cy();
        float invfx = cam_->fxinv(); float invfy = cam_->fyinv();

        // Initialize parameters
            
        // misc.

        std::vector<float> r_prev(N_*M_*2, 0.0f);
        float err = 0.f;
        float err_prev = 1e10f;
  

        for(int iter = 0; iter < MAX_ITER; ++iter){
            // Set the Parameter Vector.
            setParameterVectorFromPosesPoints();            

            // Reset A, B, Bt, C, Cinv, a, b, x, y...
            zeroizeMatrices();

            // timer::tic();
            // Iteratively solve. (Levenberg-Marquardt algorithm)
            int cnt = 0;
            for(int i = 0; i < M_; ++i){
                const Point&       Xi  = lms_ba_[i].X;
                const FramePtrVec& kfs = lms_ba_[i].kfs_seen;
                const PixelVec&    pts = lms_ba_[i].pts_on_kfs;

                for(int jj = 0; jj < pts.size(); ++jj){
                    const Pixel&   pij = pts[jj];
                    const FramePtr& kf = kfs[jj];

                    // 0) check whether it is optimizable keyframe
                    bool is_optimizable_keyframe = false;
                    int j = -1;
                    if(kfmap_optimize_.find(kf) != kfmap_optimize_.end() ) {
                        is_optimizable_keyframe = true;
                        j = kfmap_optimize_[kf]; // j_opt < N_opt_
                    }

                    // Current poses
                    const PoseSE3&  T_jw = Tjw_map_[kf];
                    const Rot3&     R_jw = T_jw.block<3,3>(0,0);
                    const Pos3&     t_jw = T_jw.block<3,1>(0,3);

                    Point Xij = R_jw*Xi + t_jw; // transform a 3D point.
               
                    // 1) Qij and Rij calculation.
                    const float& xj = Xij(0), yj = Xij(1), zj = Xij(2);
                    float invz = 1.0f/zj; float invz2 = invz*invz;
                    
                    float fxinvz      = fx*invz;
                    float fyinvz      = fy*invz;
                    float xinvz       = xj*invz;
                    float yinvz       = yj*invz;
                    float fx_xinvz2   = fxinvz*xinvz;
                    float fy_yinvz2   = fyinvz*yinvz;
                    float xinvz_yinvz = xinvz*yinvz;

                    Mat23 Rij;
                    const float& r11 = R_jw(0,0), r12 = R_jw(0,1), r13 = R_jw(0,2);
                    const float& r21 = R_jw(1,0), r22 = R_jw(1,1), r23 = R_jw(1,2);
                    const float& r31 = R_jw(2,0), r32 = R_jw(2,1), r33 = R_jw(2,2);
                    Rij << fxinvz*r11-fx_xinvz2*r31, fxinvz*r12-fx_xinvz2*r32, fxinvz*r13-fx_xinvz2*r33, 
                           fyinvz*r21-fy_yinvz2*r31, fyinvz*r22-fy_yinvz2*r32, fyinvz*r23-fy_yinvz2*r33;
                  
                    Mat26 Qij;
                    if(is_optimizable_keyframe){ // opt. frames.
                        Qij << fxinvz,0,-fx_xinvz2,-fx*xinvz_yinvz,fx*(1.f+xinvz*xinvz), -fx*yinvz,
                               0,fyinvz,-fy_yinvz2,-fy*(1.f+yinvz*yinvz),fy*xinvz_yinvz,  fy*xinvz;
                    }

                    // 2) residual calculation
                    Vec2 rij;
                    Pixel ptw;
                    ptw.x = fx*xinvz + cx;
                    ptw.y = fy*yinvz + cy;
                    rij << ptw.x - pij.x, ptw.y - pij.y;

                    // 3) HUBER weight calculation (Manhattan distance)
                    float absrxry = abs(rij(0))+abs(rij(1));
                    r_prev[cnt] = absrxry;

                    float weight = 1.0f;
                    bool flag_weight = false;
                    if(absrxry > THRES_HUBER_){
                        weight = (THRES_HUBER_/absrxry);
                        flag_weight = true;
                    }

                    // 4) Add (or fill) data (JtWJ & mJtWr & err).  
                    if(is_optimizable_keyframe) {
                        // Optimizable keyframe.
                        int j = kfmap_optimize_[kf]; // j_opt < N_opt_
                        Mat66 Qij_t_Qij = Qij.transpose()*Qij; // fixed pose, opt. pose
                        Mat63 Qij_t_Rij = Qij.transpose()*Rij; // fixed pose, opt. pose
                        Mat36 Rij_t_Qij = Qij_t_Rij.transpose(); // fixed pose, opt. pose
                        Vec6 Qij_t_rij = Qij.transpose()*rij; // fixed pose, opt. pose
                        if(flag_weight){
                            Qij_t_Qij *= weight;
                            Qij_t_Rij *= weight;
                            Rij_t_Qij = Qij_t_Rij.transpose();
                            Qij_t_rij *= weight;
                        }
                        // JtWJ(idx_pose0:idx_pose1, idx_pose0:idx_pose1)  += weight*Qij.'*Qij;
                        // JtWJ(idx_pose0:idx_pose1, idx_point0:idx_point1) = weight*Qij.'*Rij;
                        // mJtWr(idx_pose0:idx_pose1,0)   -= weight*Qij.'*rij;
                        // addData(JtWJ, Qij_t_Qij, idx_pose0, idx_pose0, 6,6);
                        // insertData(JtWJ, Qij_t_Rij, idx_pose0, idx_point0, 6,3);
                        // insertData(JtWJ, Rij_t_Qij, idx_point0, idx_pose0, 3,6);          
                        // addData(mJtWr,-Qij_t_rij, idx_pose0, 0, 6,1);
                        A_[j]    += Qij_t_Qij;
                        B_[j][i]  = Qij_t_Rij;
                        Bt_[i][j] = Rij_t_Qij;
                        a_[j]    -= Qij_t_rij;
                    } 

                    Mat33 Rij_t_Rij = Rij.transpose()*Rij; // fixed pose
                    Vec3 Rij_t_rij  = Rij.transpose()*rij; // fixed pose
                    if(flag_weight){
                        Rij_t_Rij *= weight;
                        Rij_t_rij *= weight;
                    }

                    // JtWJ(idx_point0:idx_point1,idx_point0:idx_point1) += weight*Rij.'*Rij;
                    // mJtWr(idx_point0:idx_point1,0) -= weight*Rij.'*rij;
                    // addData(JtWJ, Rij_t_Rij, idx_point0, idx_point0, 3,3);
                    // addData(mJtWr,-Rij_t_rij, idx_point0, 0, 3,1);
                    C_[i]   += Rij_t_Rij;
                    Cinv_[i] = C_[i].inverse();
                    b_[i]   -= Rij_t_rij;

                    float err_tmp = weight*rij.transpose()*rij;
                    err += err_tmp;
                
                    ++cnt;
                } // END jj
            } // END i
            // Solve problem.
            // Damping (lambda)

            // SpMat CC(3*M, 3*M);
            // CC.reserve(Eigen::VectorXi::Constant(3*M,3));
            
            // Solve! (Cholesky decomposition based solver. JtJ is sym. positive definite.)
            // timer::tic();
            // SpMat a = mJtWr.block(0,0,6*N_opt,1);
            // SpMat b = mJtWr.block(6*N_opt,0,3*M,1);
            // Eigen::SimplicialCholesky<SpMat> chol11(AA-BCinv*BB.transpose());
            // Eigen::VectorXf x = chol11.solve(a-BCinv*b);
            // Eigen::VectorXf y = CC*b-BCinv.transpose()*x;
            // Eigen::VectorXf delta_theta;
            // delta_theta << x, y;
            // timer::toc(1);
            // timer::tic();
            // Eigen::SimplicialCholesky<SpMat> chol(JtWJ);
            // Eigen::VectorXf  delta_theta = chol.solve(mJtWr);
            // // std::cout << delta_theta.transpose() <<std::endl;
            // std::cout << "chol time : " << timer::toc(0) << " [ms]" << std::endl; // 그냥 통째로 풀면 한 iteration 당 40 ms (desktop)

            // std::cout << "dimension : " << delta_theta.rows() << ", 6*N+3*M: " << 6*N+3*M << std::endl;
            
            // Update parameters (T_w2l, T_w2l_inv, xi_w2l, database->X)
            // Should omit the first image pose update. (index = 0)
            // double step_size = 1.0;
            // SpVec delta_parameter(len_parameter, 1);
            // for(int i = 0; i < delta_theta.rows(); ++i) delta_parameter.coeffRef(i,0) = delta_theta(i);
            // std::cout << "  iter: " << iter << ", meanerr: " << err/(float)cnt << " [px], delta xi: " << delta_theta.block<12,1>(0,0).norm() << std::endl;

            // parameter += delta_parameter; 

            // Update Poses and Points.
            getPosesPointsFromParameterVector();


            // std::cout << iter << "-th iter time: " << timer::toc(0) << " [ms]\n";


        } // END iter

        // Finally, update parameters
        // for(int j = NUM_FIX_KEYFRAMES; j < kfs_all.size(); ++j){
        //     kfs_all[j]->setPose(T_jw_kfs[j].inverse());
        // }
        // for(int i = 0; i < lms_ba.size(); ++i){
        //     lms_ba[i].lm->set3DPoint(lms_ba[i].X);
        // }

        // Finish
        // std::cout << "======= Local Bundle adjustment - sucess:" << (flag_success ? "SUCCESS" : "FAILED") << "=======\n";
            std::cout << "time to solve: " << timer::toc(0) << " [ms]\n";

    };

    // Reset local BA solver.
    void reset(){
        A_.resize(0);
        B_.resize(0);
        Bt_.resize(0);
        C_.resize(0);

        a_.resize(0);
        b_.resize(0);

        Cinv_.resize(0); // M blocks diagonal part for landmarks' 3D points (inverse)
        BCinv_.resize(0); // N_opt X M blocks (6x3)
        BCinvBt_.resize(0); // N_opt x N_opt blocks (6x6)
        BCinv_b_.resize(0); // N_opt x 1 blocks (6x1)
        Bt_x_.resize(0); // M x 1 blocks (3x1)

        x_.resize(0); // N_opt blocks (6x1)
        y_.resize(0); // M blocks (3x1)

        Tjw_map_.clear();
        kfmap_optimize_.clear();
        lms_ba_.resize(0);

        std::cout << "Reset bundle adjustment solver.\n";
    };

// Related to parameter vector
private:

    void setParameterVectorFromPosesPoints(){
        if(kfmap_optimize_.empty()) throw std::runtime_error("kfmap_optimize_.empty() == true.");        // initialize optimization parameter vector.
        // 1) Pose part
        for(auto kf : kfmap_optimize_){
            const FramePtr& f = kf.first;
            int j = kf.second;

            PoseSE3Tangent xi_jw;
            geometry::SE3Log_f(Tjw_map_[f],xi_jw);
            params_poses_[j] = xi_jw;
        }
        // 2) Point part
        for(int i = 0; i < M_; ++i) {
            params_points_[i] = lms_ba_[i].X;
        }
    };

    void getPosesPointsFromParameterVector(){
        // Generate parameters
        // xi part 0~5, 6~11, ... 
        int idx = 0;
        for(int j = 0; j < N_opt_; ++j){
            PoseSE3 T_jw;
            geometry::se3Exp_f(params_poses_[j], T_jw);
            Tjw_map_[kfvec_optimize_[j]] = T_jw;
        }
        // point part
        idx = 6*N_opt_;
        for(int i = 0; i < M_; ++i) lms_ba_[i].X = params_points_[i];
    };

    void zeroizeMatrices(){
        // std::cout << "in zeroize \n";
        for(int j = 0; j < N_opt_; ++j){
            A_[j].setZero();
            // std::cout << "A:\n" << A_[j] << std::endl;
            a_[j].setZero();
            // std::cout << "A:\n" << a_[j] << std::endl;

            x_[j].setZero();
            // std::cout << "A:\n" << x_[j] << std::endl;

            params_poses_[j].setZero();
            // std::cout << "A:\n" << params_poses_[j] << std::endl;

            BCinv_b_[j].setZero();
            // std::cout << "A:\n" << BCinv_b_[j] << std::endl;
            for(int i = 0; i < M_; ++i){
                B_[j][i].setZero();
                Bt_[i][j].setZero();
                BCinv_[j][i].setZero();
            }
            for(int i = 0; i < N_opt_;++i){
                BCinvBt_[j][i].setZero();
            }
        }
        for(int i = 0; i < M_; ++i){
            C_[i].setZero();
            Cinv_[i].setZero();
            b_[i].setZero();
            y_[i].setZero();
            params_points_[i].setZero();
            Bt_x_[i].setZero();
        }
    };

// Add and insert functions
private:

    void insertToPoseBlock(const Mat66& mat, int num_block){
        if(num_block < 0 || num_block >= N_opt_) throw std::runtime_error("num_block < 0 || num_block >= N_opt_");
        A_[num_block] = mat;
    };

    void addToPoseBlock(const Mat66& mat, int num_block){
        if(num_block < 0 || num_block >= N_opt_) throw std::runtime_error("num_block < 0 || num_block >= N_opt_");
        A_[num_block] += mat;
    };

    void insertToLandmarkBlock(const Mat33& mat, int num_block){
        if(num_block < 0 || num_block >= M_) throw std::runtime_error("num_block < 0 || num_block >= M_");
        C_[num_block] = mat;
    };

    void addToLandmarkBlock(const Mat33& mat, int num_block){
        if(num_block < 0 || num_block >= M_) throw std::runtime_error("num_block < 0 || num_block >= M_");
        C_[num_block] += mat;
    };

    void insertToPoseLandmarkBlock(const Mat63& mat, int n_pose, int n_landmark){

    };

private:
    void inverseLandmarkBlockDiagonals(){
        for(auto& mat : C_) mat = mat.inverse();
    };


    
};
#endif