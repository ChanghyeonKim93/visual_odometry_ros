#ifndef _SPARSE_BUNDLE_ADJUSTMENT_SCALE_SQP_H_
#define _SPARSE_BUNDLE_ADJUSTMENT_SCALE_SQP_H_

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

#include "core/ba_solver/ba_types.h"
#include "core/ba_solver/landmark_ba.h"
#include "core/ba_solver/sparse_ba_parameters.h"
#include "core/scale_estimator/scale_constraint.h"
#include "core/ba_solver/connectivity_map.h"

#define N_OPT_PRIOR 500
#define M_PRIOR 50000
#define K_PRIOR 1000

struct LandmarkBA;
class SparseBAParameters;
class SparseBundleAdjustmentScaleSQPSolver;

/*
    <Problem we want to solve (SQP with Equality Constraint for scale constraint on the translation motion only)>
       
        H*delta_theta = J.transpose()*r;

    where
    - Hessian  
            H = [ A_ ,  B_,  Dt_
                  Bt_,  C_,  0 
                  D_ ,  0 ,  0  ];

    - Jacobian multiplied by residual vector 
            J.transpose()*r = [a;b;c];

    - Update parameters
            delta_theta = [x;y;z];

    - Solutions
    z = [ D*(A - B*Cinv*Bt)^-1*Dt]^-1*[ D*(A - B*Cinv*Bt)^-1*(a - B*Cinv*b) - c ]
    x = (A - B*Cinv*Bt)^-1 * (a - B*Cinv*b - Dt*z)
    y = Cinv*( b - Cinv*Bt*x )
*/

class SparseBundleAdjustmentScaleSQPSolver
{
private:
    std::shared_ptr<Camera> cam_;

private:
    // Storages to solve Schur Complement
/*
- Hessian  
    H = [ A_ ,  B_,  Dt_
          Bt_,  C_,  0 
          D_ ,  0 ,  0  ]; \in R^{ (3*N_opt + 3*M + K) x (3*N_opt + 3*M + K) }

- RHS = [ a_
          b_
          c_]; \in R^{ (3*N_opt + 3*M + K) x 1 }
*/

/*
< Notation Rule >

    j : pose index
    i : point index
    k : constraint index

*/

// Base storages
    DiagBlockMat33 A_;  // j    N_opt (3x3) block diagonal part for poses (translation only)
    FullBlockMat33 B_;  // j,i  N_opt x M (3x3) block part (side)
    FullBlockMat33 Bt_; // i,j  M x N_opt (3x3) block part (side, transposed)
    DiagBlockMat33 C_;  // i    M (3x3) block diagonal part for landmarks' 3D points
    FullBlockMat31 Dt_; // j,k  N_opt x K (3x1) block part
    FullBlockMat13 D_;  // k,j  K x N_opt (1x3) block part

    BlockVec3 a_; // j N_opt x 1 (3x1) (the number of optimization poses == N-1)
    BlockVec3 b_; // i     M x 1 (3x1) (the number of landmarks)
    BlockVec1 c_; // k     K x 1 (1x1) (the number of constrained poses (turning frames))

// Update step
    BlockVec3 x_; // j N_opt (3x1) translation
    BlockVec3 y_; // i    M (3x1) landmarks
    BlockVec1 z_; // k    K (1x1) Lagrange multiplier

// parameters to be considered.
    DiagBlockMat33 fixparams_rot_;   // N_opt (3x3) rotation (fixed!). (R_jw)

    BlockVec3 params_trans_;         // N_opt (3x1) 3D translation of the frame. (t_jw)
    BlockVec3 params_points_;        //     M (3x1) 3D points (Xi)
    BlockVec1 params_lagrange_;      //     K (1x1) Lagrange multiplier (lambda_k)

// Derivated Storages
    DiagBlockMat33 Cinv_;    // i    M (3x3) block diag mat.
    BlockVec3      Cinvb_;   // i    M  (3x1) block vector
    FullBlockMat33 BCinv_;   // j,i  N_opt x M (3x3) block full mat.
    BlockVec3      BCinvb_;  // j    N_opt x 1 (3x1) block vector
    FullBlockMat33 BCinvBt_; // j,j  N_opt x N_opt (3x3) block full mat.

    FullBlockMat33 Ap_; // j,j  N_opt x N_opt (3x3), == A - B*Cinv*Bt
    BlockVec3      ap_; // j    N_opt x 1 (3x1) == a - B*Cinv*b

    BlockVec3      Apinv_ap_; // j    N_opt x 1 (3x1) == (A - B*Cinv*Bt)^-1 * (a - B*Cinv*b)
    FullBlockMat31 Apinv_Dt_; // j,k  N_opt x K (3x1) == (A - B*Cinv*Bt)^-1 * Dt
    
    BlockVec1      D_Apinv_ap_; // k    K x 1 (1x1)
    FullBlockMat11 D_Apinv_Dt_; // k,k  K x K (1x1)


// Large Eigen matrix
    _BA_MatX Ap_mat_;       // 3*N_opt x 3*N_opt
    _BA_MatX ap_mat_;       // 3*N_opt x       1
    _BA_MatX D_mat_;        // K       x 3*N_opt
    _BA_MatX Dt_mat_;       // 3*N_opt x       K
    _BA_MatX a_mat_;        // 3*N_opt x       1

    _BA_MatX Apinv_ap_mat_; // 3*N_opt x       1
    _BA_MatX Apinv_Dt_mat_; // 3*N_opt x       K

    _BA_MatX D_Apinv_Dt_mat_;     // K       x       K
    _BA_MatX D_Apinv_ap_m_c_mat_; // K       x       1

    _BA_MatX z_mat_; // K       x       1
    _BA_MatX x_mat_; // 3*N_opt x       1



    // Input variable
    std::shared_ptr<SparseBAParameters> ba_params_;
    std::shared_ptr<ScaleConstraints>   scale_const_;
    
    // Connectivity map
    // ij, ji (i는 여러 j와 연관되어있다. )
    // kj, jk

private:
    int N_;     // # of total poses including fixed poses
    int N_opt_; // # of poses to be optimized
    int M_;     // # of landmarks to be optimized
    int K_;     // # of equality constraints
    int n_obs_; // # of total observations ( == length of residual vector / 2)

private:
    // Optimization parameters
    _BA_numeric THRES_HUBER_;
    _BA_numeric THRES_EPS_;

public:
    /// @brief Constructor for BundleAdjustmentSolver ( trans (3-DoF), 3D points (3-DoF) ). Sparse solver with 3x3 block diagonals, 3x3 block diagonals, 3x3 blocks.
    SparseBundleAdjustmentScaleSQPSolver();

    /// @brief Set BA parameters and Constraints
    /// @param ba_params ba paramseters (poses, landmarks)
    /// @param scale_const (scale constraints)
    void setBAParametersAndConstraints(const std::shared_ptr<SparseBAParameters>& ba_params, const std::shared_ptr<ScaleConstraints>& scale_const);

    void setHuberThreshold(float thres_huber);
    void setCamera(const std::shared_ptr<Camera>& cam);
    
// Solve and reset.
public:
    /// @brief Solve the BA for fixed number of iterations
    /// @param MAX_ITER Maximum iterations
    /// @return true when success, false when failed.
    bool solveForFiniteIterations(int MAX_ITER);
    
    /// @brief Reset the BA solver.
    void reset();

// Set the problem size
private:
    void makeStorageSizeToFit();  

private:
    void zeroizeStorageMatrices();
    void setParameterVectorFromPosesPoints();
    void initializeLagrangeMultipliers();
    void getPosesPointsFromParameterVector();

private:
    inline void calc_Rij_t_Rij(const _BA_Mat23& Rij,
        _BA_Mat33& Rij_t_Rij);
    inline void calc_Rij_t_Rij_weight(const _BA_numeric weight, const _BA_Mat23& Rij,
        _BA_Mat33& Rij_t_Rij);

    inline void calc_Qij_t_Qij(const _BA_Mat23& Qij, 
        _BA_Mat33& Qij_t_Qij);
    inline void calc_Qij_t_Qij_weight(const _BA_numeric weight, const _BA_Mat23& Qij, 
        _BA_Mat33& Qij_t_Qij);
};
#endif