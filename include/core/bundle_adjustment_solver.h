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
        Hessian pattern
        H = [Hpp_, Hpl_;
            Hlp_, Hll_];
    */
    BlockDiagonalMat66 Hpp_; // N_opt (6x6) block diagonal part for poses
    BlockFullMat63     Hpl_; // N_opt x M (6x3) block part (side)
    BlockFullMat36     Hlp_; // M x N_opt (3x6) block part (side, transposed)
    BlockDiagonalMat33 Hll_; // M (3x3) block diagonal part for landmarks' 3D points
    
    BlockVec6 a_; // N_opt x 1 (6x1) 
    BlockVec3 b_; // M x 1 (3x1)

    // Variables to solve Schur Complement
    BlockDiagonalMat33 Hll_inv_; // block diagonal part for landmarks' 3D points (inverse)
    BlockFullMat63     HplHll_inv_; // N_opt X M  (6x3)
    BlockFullMat66     HplHll_inv_Hpl_t_; // N_opt x N_opt (6x6)
    BlockVec6          HplHll_inv_b_; // N_opt x 1 (6x1)
    BlockVec3          Hpl_t_dx_; // M x 1 (3x1)

    BlockVec6 dx_; // N_opt blocks (6x1)
    BlockVec3 dy_; // M blocks (3x1)

// Variables and problem size
    LandmarkBAVec lms_ba_;
    PoseSE3Vec    T_jw_;

    int N_;     // # of total poses including fixed poses
    int N_opt_; // # of poses to be optimized
    int M_;     // # of landmarks to be optimized

public:
    BundleAdjustmentSolver() {
        // reserve memory
        Hpp_.reserve(200);
        Hpl_.reserve(200);
        Hlp_.reserve(100000);
        Hll_.reserve(100000);
    };

    // N:     # of total poses (including fixed poses)
    // N_opt: # of optimizable poses
    // M:     # of landmarks to be optimized
    void setProblemSize(int N, int N_opt, int M){ 
        N_     = N; 
        N_opt_ = N_opt;
        M_     = M;

        Hpp_.resize(N_opt_); 
        Hpl_.resize(N_opt_); for(auto v : Hpl_) v.resize(M_);
        Hlp_.resize(M_); for(auto v : Hlp_) v.resize(N_opt_);
        Hll_.resize(M_);   

        a_.resize(N_opt_); // 6*N_opt x 1 
        b_.resize(M_); // 3*M x M_

        Hll_inv_.resize(M_); // M blocks diagonal part for landmarks' 3D points (inverse)
        HplHll_inv_.resize(N_opt_, std::vector<Mat63>(M_)); // N_opt X M blocks (6x3)
        HplHll_inv_Hpl_t_.resize(N_opt_, std::vector<Mat66>(N_opt_)); // N_opt x N_opt blocks (6x6)
        HplHll_inv_b_.resize(N_opt_); // N_opt x 1 blocks (6x1)
        Hpl_t_dx_.resize(M_); // M x 1 blocks (3x1)

        dx_.resize(N_opt_); // N_opt blocks (6x1)
        dy_.resize(M_); // M blocks (3x1)
    };

    void setVariables(const PoseSE3Vec& T_jw_kfs, const LandmarkBAVec& lms_ba){
        // observation 숫자를 셀 필요자체가 없다 ....
        T_jw_ = T_jw_kfs;
        lms_ba_.resize(lms_ba.size());
        for(int i = 0; i < lms_ba.size(); ++i) lms_ba_[i] = lms_ba[i];
    };

    void reset(){
        Hpp_.resize(0);
        Hpl_.resize(0);
        Hlp_.resize(0);
        Hll_.resize(0);

        a_.resize(0); // 6*N_opt x 1 
        b_.resize(0); // 3*M x M_

        Hll_inv_.resize(0); // M blocks diagonal part for landmarks' 3D points (inverse)
        HplHll_inv_.resize(0); // N_opt X M blocks (6x3)
        HplHll_inv_Hpl_t_.resize(0); // N_opt x N_opt blocks (6x6)
        HplHll_inv_b_.resize(0); // N_opt x 1 blocks (6x1)
        Hpl_t_dx_.resize(0); // M x 1 blocks (3x1)

        dx_.resize(0); // N_opt blocks (6x1)
        dy_.resize(0); // M blocks (3x1)

        std::cout << "Reset bundle adjustment solver.\n";
    };


    // Add and insert functions
    void insertToPoseBlock(const Mat66& mat, int num_block){
        if(num_block < 0 || num_block >= N_opt_) throw std::runtime_error("num_block < 0 || num_block >= N_opt_");
        Hpp_[num_block] = mat;
    };

    void addToPoseBlock(const Mat66& mat, int num_block){
        if(num_block < 0 || num_block >= N_opt_) throw std::runtime_error("num_block < 0 || num_block >= N_opt_");
        Hpp_[num_block] += mat;
    };

    void insertToLandmarkBlock(const Mat33& mat, int num_block){
        if(num_block < 0 || num_block >= M_) throw std::runtime_error("num_block < 0 || num_block >= M_");
        Hll_[num_block] = mat;
    };

    void addToLandmarkBlock(const Mat33& mat, int num_block){
        if(num_block < 0 || num_block >= M_) throw std::runtime_error("num_block < 0 || num_block >= M_");
        Hll_[num_block] += mat;
    };

    void insertToPoseLandmarkBlock(const Mat63& mat, int n_pose, int n_landmark){

    };

private:
    void inverseLandmarkBlockDiagonals(){
        for(auto& mat : Hll_) mat = mat.inverse();
    };


    
};
#endif