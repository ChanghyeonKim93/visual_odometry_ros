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

// BA related headers
#include "core/ba_solver/ba_types.h"
#include "core/ba_solver/landmark_ba.h"
#include "core/ba_solver/sparse_ba_parameters.h"


struct LandmarkBA;
class SparseBAParameters;
class SparseBundleAdjustmentSolver;

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

/// @brief A sparse solver for a feature-based Bundle adjustment problem.
class SparseBundleAdjustmentSolver
{
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

    // Input variable
    std::shared_ptr<SparseBAParameters> ba_params_;

private:
    // Problem sizes
    int N_;     // # of total poses including fixed poses
    int N_opt_; // # of poses to be optimized
    int M_;     // # of landmarks to be optimized
    int n_obs_;

private:
    // Optimization parameters
    _BA_numeric THRES_HUBER_;
    _BA_numeric THRES_EPS_;

public:
    /// @brief Constructor for BundleAdjustmentSolver ( se3 (6-DoF), 3D points (3-DoF) ). Sparse solver with 6x6 block diagonals, 3x3 block diagonals, 6x3 and 3x6 blocks.
    SparseBundleAdjustmentSolver();

    /// @brief Set connectivities, variables...
    /// @param ba_params bundle adjustment parameters
    void setBAParameters(const std::shared_ptr<SparseBAParameters>& ba_params);

    /// @brief Set Huber threshold for SLBA
    /// @param thres_huber Huber norm threshold value
    void setHuberThreshold(_BA_numeric thres_huber);

    /// @brief Set camera pointer
    /// @param cam camera pointer
    void setCamera(const std::shared_ptr<Camera>& cam);

    /// @brief Solve the BA for fixed number of iterations
    /// @param MAX_ITER Maximum iterations
    /// @return true when success, false when failed.
    bool solveForFiniteIterations(int MAX_ITER);

    /// @brief Reset local BA solver.
    void reset();

private:
    /// @brief Set problem sizes and resize the storages.
    /// @param N the number of poses (including fixed & optimizable poses)
    /// @param N_opt the number of optimizable poses (N_opt < N)
    /// @param M the number of landmarks
    /// @param n_obs the number of observations
    void setProblemSize(int N, int N_opt, int M, int n_obs);

// Related to parameter vector
private:
    void setParameterVectorFromPosesPoints();
    void getPosesPointsFromParameterVector();
    void zeroizeStorageMatrices();    

// For fast calculations for symmetric matrices
private:
    inline void calc_Rij_t_Rij(const _BA_Mat23& Rij,
        _BA_Mat33& Rij_t_Rij);
    inline void calc_Rij_t_Rij_weight(const _BA_numeric weight, const _BA_Mat23& Rij,
        _BA_Mat33& Rij_t_Rij);

    inline void calc_Qij_t_Qij(const _BA_Mat26& Qij, 
        _BA_Mat66& Qij_t_Qij);
    inline void calc_Qij_t_Qij_weight(const _BA_numeric weight, const _BA_Mat26& Qij, 
        _BA_Mat66& Qij_t_Qij);


};
#endif