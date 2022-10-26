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
#include "util/cout_color.h"

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
    bool is_stereo_mode_;

private:
    std::vector<std::shared_ptr<Camera>> cams_;
    int n_cams_;

private:
    // Storages to solve Schur Complement
    DiagBlockMat66 A_; // N_opt (6x6) block diagonal part for poses +
    FullBlockMat63 B_; // N_opt x M (6x3) block part (side) +
    FullBlockMat36 Bt_; // M x N_opt (3x6) block part (side, transposed) +
    DiagBlockMat33 C_; // M (3x3) block diagonal part for landmarks' 3D points +
    
    BlockVec6 a_; // N_opt x 1 (6x1) +
    BlockVec3 b_; // M x 1 (3x1) +

    BlockVec6 x_; // N_opt  (6x1)+
    BlockVec3 y_; // M  (3x1) +

    BlockVec6 params_poses_;  // N_opt (6x1) parameter vector for poses
    BlockVec3 params_points_; // M     (3x1) parameter vector for points

    DiagBlockMat33 Cinv_;    // M (3x3) block diagonal part for landmarks' 3D points (inverse) +
    FullBlockMat63 BCinv_;   // N_opt X M  (6x3) +
    FullBlockMat36 CinvBt_;  // M x N_opt (3x6) +
    FullBlockMat66 BCinvBt_; // N_opt x N_opt (6x6) +
    BlockVec6      BCinv_b_; // N_opt (6x1) +
    BlockVec3      Bt_x_;    // M     (3x1) +
    BlockVec3      Cinv_b_;  // M     (3x1) +

    FullBlockMat66 Am_BCinvBt_; // N_opt x N_opt (6x6) +
    BlockVec6      am_BCinv_b_; // N_opt (6x1) +
    BlockVec3      CinvBt_x_;   // M (3x1) +

// Dynamic matrix
    _BA_MatX Am_BCinvBt_mat_; // 6*N_opt_, 6*N_opt_
    _BA_MatX am_BCinv_b_mat_;  // 6*N_opt_, 1

    _BA_MatX x_mat_; // 6*N_opt_, 1

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
    _BA_Numeric THRES_HUBER_;
    _BA_Numeric THRES_EPS_;

public:
    /// @brief Constructor for BundleAdjustmentSolver ( se3 (6-DoF), 3D points (3-DoF) ). Sparse solver with 6x6 block diagonals, 3x3 block diagonals, 6x3 and 3x6 blocks.
    SparseBundleAdjustmentSolver(bool is_stereo = false);

    /// @brief Set connectivities, variables...
    /// @param ba_params bundle adjustment parameters
    void setBAParameters(const std::shared_ptr<SparseBAParameters>& ba_params);

    /// @brief Set Huber threshold for SLBA
    /// @param thres_huber Huber norm threshold value
    void setHuberThreshold(_BA_Numeric thres_huber);

    /// @brief Set camera pointer
    /// @param cam camera pointer
    void setCamera(const std::shared_ptr<Camera>& cam);

    /// @brief Set stereo camera pointers. Before call this function, 'is_stereo' should be set to 'true'.
    /// @param cam0 camera pointer (left)
    /// @param cam1 camera pointer (right)
    void setStereoCameras(const std::shared_ptr<Camera>& cam0, const std::shared_ptr<Camera>& cam1);

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
    inline void calc_Rij_t_Rij_weight(const _BA_Numeric weight, const _BA_Mat23& Rij,
        _BA_Mat33& Rij_t_Rij);

    inline void calc_Qij_t_Qij(const _BA_Mat26& Qij, 
        _BA_Mat66& Qij_t_Qij);
    inline void calc_Qij_t_Qij_weight(const _BA_Numeric weight, const _BA_Mat26& Qij, 
        _BA_Mat66& Qij_t_Qij);


};
#endif