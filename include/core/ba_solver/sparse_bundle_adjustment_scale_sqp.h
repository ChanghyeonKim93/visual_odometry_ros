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
    < Governed by 'i' >
    C_  [ i ] = Sum(Rij.'*Rij)
    b_  [ i ] = Sum(Rij.'*rij) + additional
    
    for(int i = 0; i < M_; ++i)
        Cinv_b_[i] = Cinv_[i]*b_[i];  // FILL STORAGE (10)


    < Governed by 'j' >
    A_  [ j ] = sum(Qij.'*Qij)
    a_  [ j ] = sum(Qij.'*rij) + additional


    < Governed by 'i, j' >
    B_  [ j, i ]
    Bt_ [ i, j ]


    < Governed by 'k, j' >
    D_  [ k, j ] 
    Dt_ [ j, k ]

*/  

/*
    Connectivity map

    i <-> j
    j <-> k
    
*/

// Base storages
    BlockDiagMat33 A_;  // j    N_opt (3x3) block diagonal part for poses (translation only)
    BlockFullMat33 B_;  // j,i  N_opt x M (3x3) block part (side)
    BlockFullMat33 Bt_; // i,j  M x N_opt (3x3) block part (side, transposed)
    BlockDiagMat33 C_;  // i    M (3x3) block diagonal part for landmarks' 3D points
    BlockFullMat31 Dt_; // i,k  N_opt x K (3x1) block part
    BlockFullMat13 D_;  // k,i  K x N_opt (1x3) block part

    BlockVec3 a_; // j N_opt x 1 (3x1) (the number of optimization poses == N-1)
    BlockVec3 b_; // i     M x 1 (3x1) (the number of landmarks)
    BlockVec1 c_; // k     K x 1 (1x1) (the number of constrained poses (turning frames))

// Update step
    BlockVec3 x_; // N_opt (3x1) translation
    BlockVec3 y_; //     M (3x1) landmarks
    BlockVec1 z_; //     K (1x1) Lagrange multiplier

// parameters to be considered.
    BlockDiagMat33 fixparams_rot_;   // N_opt (3x3) rotation (fixed!). (R_jw)

    BlockVec3 params_trans_;         // N_opt (3x1) 3D translation of the frame. (t_jw)
    BlockVec3 params_points_;        //     M (3x1) 3D points (Xi)
    BlockVec1 params_lagrange_;      //     K (1x1) Lagrange multiplier (lambda_k)

// Derivated Storages
    BlockDiagMat33 Cinv_;    // M (3x3) block diag
    BlockVec3      Cinvb_;   // M x 1 (3x1) block vector
    BlockFullMat33 CinvBt_;  // M x N_opt (3x3) block full mat.
    BlockVec3      CinvBtx_; // M x 1 (3x1) block vector
    BlockFullMat33 BCinv_;   // N_opt x M (3x3) block full mat.
    BlockVec3      BCinvb_;  // N_opt x 1 (3x1) block vector
    BlockFullMat33 BCinvBt_; // N_opt x N_opt (3x3) block full mat.

    BlockFullMat33 Am_BCinvBt_;         // N_opt x N_opt (3x3) block full mat.
    BlockFullMat33 inv_Am_BCinvBt_;     // N_opt x N_opt (3x3) block full mat.
    BlockVec3      am_BCinvb_;          // N_opt x 1 (3x1) block vector
    BlockVec3      am_BCinvbm_Dtz_;     // N_opt x 1 (3x1) block vector

    BlockFullMat11 D_inv_Am_BCinvBt_Dt_; // K fx K (1x1) full mat.

    


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
};
#endif