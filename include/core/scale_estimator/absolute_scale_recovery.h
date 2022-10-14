#ifndef _ABSOLUTE_SCALE_RECOVERY_H_
#define _ABSOLUTE_SCALE_RECOVERY_H_

#include <iostream>
#include <string>
#include <exception>
#include <numeric>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>
#include <Eigen/SparseQR>

#include <opencv2/core.hpp>

#include "core/type_defines.h"
#include "core/defines.h"

#include "core/camera.h"
#include "core/mapping.h"
#include "core/landmark.h"
#include "core/frame.h"

#include "util/timer.h"

// SQP with eq. constraints solver
#include "core/scale_estimator/scale_constraint.h"
#include "core/ba_solver/sparse_bundle_adjustment_scale_sqp.h"

/// @brief Absolute Scale Recovery Class (ASR)
class AbsoluteScaleRecovery
{
private:
    std::shared_ptr<Camera> cam_;

private:
    std::shared_ptr<SparseBundleAdjustmentScaleSQPSolver> sqp_solver_;

public:
    /// @brief ASR constructor (with camera pointer)
    /// @param cam 
    AbsoluteScaleRecovery(const std::shared_ptr<Camera>& cam);
    
    /// @brief ASR destructor
    ~AbsoluteScaleRecovery();

public:
    /// @brief Run Absolute Scale Recovery
    /// @param frames_t0 turning frames (previous)
    /// @param frames_u unconstrained frames between two turning regions
    /// @param frames_t1 turning frames (current)
    void runASR(
        const FramePtrVec& frames_t0, 
        const FramePtrVec& frames_u, 
        const FramePtrVec& frames_t1);

};

#endif