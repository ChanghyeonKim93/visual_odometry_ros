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
          D_ ,  0 ,  0  ]; \in R^{ (3*N_opt + 3*M + N_t) x (3*N_opt + 3*M + N_t) }
*/
    BlockDiagMat33 A_; // N_opt (3x3) block diagonal part for poses (translation only)
    BlockFullMat33 B_; // N_opt x M (3x3) block part (side)
    BlockFullMat33 Bt_; // M x N_opt (3x3) block part (side, transposed)
    BlockDiagMat33 C_; // M (3x3) block diagonal part for landmarks' 3D points
    BlockFullMat13 D_; // Nt x N_opt (1x3) block part
    BlockFullMat31 Dt_; // N_opt x Nt (3x1) block part

/*
- RHS = [ a_
          b_
          c_]; \in R^{ (3*N_opt + 3*M + N_t) x 1 }
*/
    BlockVec3 a_; // N_opt x 1 (3x1) (the number of optimization poses == N-1)
    BlockVec3 b_; // M x 1     (3x1) (the number of landmarks)
    BlockVec1 c_; // Nt x 1    (1x1) (the number of constrained poses (turning frames))


private:
    // Optimization parameters
    _BA_numeric THRES_HUBER_;
    _BA_numeric THRES_EPS_;

public:
    /// @brief Constructor for BundleAdjustmentSolver ( trans (3-DoF), 3D points (3-DoF) ). Sparse solver with 3x3 block diagonals, 3x3 block diagonals, 3x3 blocks.
    SparseBundleAdjustmentScaleSQPSolver();

    void setBAParameters();
    void setHuberThreshold();
    void setCamera();
    
    
    bool solveForFiniteIterations(int MAX_ITER);

    void reset();

private:
    void setProblemSize(int N, int Nt, int M, int n_obs);

private:
    void setParameterVectorFromPosesPoints();
    void getPosesPointsFromParameterVector();
    void zeroizeStorageMatrices();
};
#endif