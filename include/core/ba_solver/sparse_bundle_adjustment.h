#ifndef _SPARSE_BUNDLE_ADJUSTMENT_H_
#define _SPARSE_BUNDLE_ADJUSTMENT_H_

#include <iostream>
#include <vector>
#include <map>
#include <set>

#include <Eigen/Dense>

#include "core/camera.h"
#include "core/landmark.h"
#include "core/frame.h"
#include "core/type_defines.h"

#include "util/geometry_library.h"
#include "util/timer.h"

#include "core/ba_solver/ba_parameters.h"

typedef std::vector<Mat66>              BlockDiagMat66; 
typedef std::vector<Mat33>              BlockDiagMat33; 
typedef std::vector<std::vector<Mat63>> BlockFullMat63; 
typedef std::vector<std::vector<Mat36>> BlockFullMat36;
typedef std::vector<std::vector<Mat66>> BlockFullMat66;
typedef std::vector<Vec6>               BlockVec6;
typedef std::vector<Vec3>               BlockVec3;

/*
    <Problem we want to solve>
        H*delta_theta = J.transpose()*r;

    where
    - Hessian  
            H = [A_,  B_;
                 Bt_, C_];

    - Jacobian multiplied by residual vector 
            J.transpose()*r = [a;b];

    - Update parameters
            delta_theta = [x;y];
*/

// A sparse solver for a feature-based Bundle adjustment problem.
class SparseBundleAdjustmentSolver{
private:
    std::shared_ptr<Camera> cam_;

private:
    // Storages to solve Schur Complement
    BlockDiagMat66 A_; // N_opt (6x6) block diagonal part for poses +
    BlockFullMat63 B_; // N_opt x M (6x3) block part (side) +
    BlockFullMat36 Bt_; // M x N_opt (3x6) block part (side, transposed) +
    BlockDiagMat33 C_; // M (3x3) block diagonal part for landmarks' 3D points +
    
    BlockVec6 a_; // N_opt x 1 (6x1) +
    BlockVec3 b_; // M x 1 (3x1) +

    BlockVec6 x_; // N_opt  (6x1)+
    BlockVec3 y_; // M  (3x1) +

    BlockVec6 params_poses_;  // N_opt (6x1) parameter vector for poses
    BlockVec3 params_points_; // M     (3x1) parameter vector for points

    BlockDiagMat33 Cinv_;    // M (3x3) block diagonal part for landmarks' 3D points (inverse) +
    BlockFullMat63 BCinv_;   // N_opt X M  (6x3) +
    BlockFullMat36 CinvBt_;  // M x N_opt (3x6) +
    BlockFullMat66 BCinvBt_; // N_opt x N_opt (6x6) +
    BlockVec6      BCinv_b_; // N_opt (6x1) +
    BlockVec3      Bt_x_;    // M     (3x1) +
    BlockVec3      Cinv_b_;  // M     (3x1) +

    BlockFullMat66 Am_BCinvBt_; // N_opt x N_opt (6x6) +
    BlockVec6      am_BCinv_b_;      // N_opt (6x1) +
    BlockVec3      CinvBt_x_;        // M (3x1) +

    // Input variables  
    LandmarkBAVec lms_ba_; // landmarks to be optimized
    std::map<FramePtr,PoseSE3> Tjw_map_; // map containing poses (for all keyframes including fixed ones)
    std::map<FramePtr,int> kfmap_optimize_; // map containing optimizable keyframes and their indexes
    std::vector<FramePtr> kfvec_optimize_;

private:
    // Problem sizes
    int N_;     // # of total poses including fixed poses
    int N_opt_; // # of poses to be optimized
    int M_;     // # of landmarks to be optimized
    int n_obs_;

private:
    // Optimization parameters
    float THRES_HUBER_;
    float THRES_EPS_;

public:
    // Constructor for BundleAdjustmentSolver ( se3 (6-DoF), 3D points (3-DoF) )
    // Sparse solver with 6x6 block diagonals, 3x3 block diagonals, 6x3 and 3x6 blocks.
    SparseBundleAdjustmentSolver();

    // Set Huber threshold
    void setHuberThreshold(float thres_huber);

    // Set camera.
    void setCamera(const std::shared_ptr<Camera>& cam);

    // Set problem sizes and resize the storages.
    void setProblemSize(int N, int N_opt, int M, int n_obs);

    // Set Input Values.
    void setInitialValues(
        const std::map<FramePtr,PoseSE3>& Tjw_map,
        const LandmarkBAVec& lms_ba,
        const std::map<FramePtr,int>& kfmap_optimize);

    // Solve the BA for fixed number of iterations
    void solveForFiniteIterations(int MAX_ITER);

    // Reset local BA solver.
    void reset();

// Related to parameter vector
private:

    void setParameterVectorFromPosesPoints();
    void getPosesPointsFromParameterVector();
    void zeroizeStorageMatrices();    
};
#endif