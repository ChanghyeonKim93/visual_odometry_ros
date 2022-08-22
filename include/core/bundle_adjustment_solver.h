#ifndef _BUNDLE_ADJUSTMENT_SOLVER_H_
#define _BUNDLE_ADJUSTMENT_SOLVER_H_

#include <iostream>
#include <vector>

#include <Eigen/Dense>
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

    // Variables
    LandmarkBAVec lms_ba_; // landmarks to be optimized
    std::map<FramePtr,PoseSE3> Tjw_map_; // map containing Tjw_map_
    std::map<FramePtr,int> kfmap_optimize_; // map containing optimizable keyframes and their indexes

    // Problem size
    int N_;     // # of total poses including fixed poses
    int N_opt_; // # of poses to be optimized
    int M_;     // # of landmarks to be optimized

public:
    // Constructor for BundleAdjustmentSolver ( se3 (6-DoF), 3D points (3-DoF) )
    // Sparse solver with 6x6 block diagonals, 3x3 block diagonals, 6x3 and 3x6 blocks.
    BundleAdjustmentSolver() {
        A_.reserve(500); // reserve expected # of optimizable poses (N_opt)
        B_.reserve(500); // 
        Bt_.reserve(1000000);
        C_.reserve(1000000); // reserve expected # of optimizable landmarks (M)
    };

    // N    : # of total poses (including fixed poses)
    // N_opt: # of optimizable poses
    // M    : # of optimizable landmarks
    void setProblemSize(int N, int N_opt, int M){ 
        N_     = N; 
        N_opt_ = N_opt;
        M_     = M;

        A_.resize(N_opt_); 
        B_.resize(N_opt_); for(auto v : B_) v.resize(M_);
        Bt_.resize(M_);    for(auto v : Bt_) v.resize(N_opt_);
        C_.resize(M_);   

        a_.resize(N_opt_); // 6*N_opt x 1 
        b_.resize(M_); // 3*M x 1

        Cinv_.resize(M_); // 3x3, M diagonal blocks 
        BCinv_.resize(N_opt_, std::vector<Mat63>(M_));       // 6x3, N_opt X M blocks
        BCinvBt_.resize(N_opt_, std::vector<Mat66>(N_opt_)); // 6x6, N_opt x N_opt blocks
        BCinv_b_.resize(N_opt_); // 6x1, N_opt x 1 blocks
        Bt_x_.resize(M_);        // 3x1, M x 1 blocks

        x_.resize(N_opt_); // 6x1, N_opt blocks
        y_.resize(M_);     // 3x1, M blocks
    };

    void setInitialValues(
        const std::map<FramePtr,PoseSE3>& Tjw_map,
        const LandmarkBAVec& lms_ba,
        const std::map<FramePtr,int>& kfmap_optimize)
    {
        Tjw_map_ = Tjw_map;
        kfmap_optimize_ = kfmap_optimize;
        lms_ba_.resize(lms_ba.size());
        for(int i = 0; i < lms_ba.size(); ++i) lms_ba_[i] = lms_ba[i];

        if(M_ != lms_ba.size()) throw std::runtime_error("In BundleAdjustmentSolver, 'M_ != lms_ba.size()'.");
        if(N_ != Tjw_map_.size()) throw std::runtime_error("In BundleAdjustmentSolver, 'N_ != Tjw_map_.size()'.");
    };

    // Solve
    void solveInFiniteIterations(int MAX_ITER){
        
        // Initialize parameters


        // Iteratively solve. (Levenberg-Marquardt algorithm)
        for(int i = 0; i < M_; ++i){
            const Point& Xi = lms_ba_[i].X;
            const FramePtrVec& kfs_seen = lms_ba_[i].kfs_seen;
            const PixelVec& pts_seen = lms_ba_[i].pts_on_kfs;

            for(int jj = 0; jj < pts_seen.size(); ++jj){
                const Pixel& pt_seen    = pts_seen[jj];
                const FramePtr& kf_seen = kfs_seen[jj];

                // Check whether this keyframe is an optimizable or not.
                if(kfmap_optimize_.find(kf_seen) != kfmap_optimize_.end() ) {
                    // Optimizable keyframe.
                    int j = kfmap_optimize_[kf_seen]; // j_opt < N_opt_
                }
                else {
                    // Fixed keyframe
                }

            }
        }  
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