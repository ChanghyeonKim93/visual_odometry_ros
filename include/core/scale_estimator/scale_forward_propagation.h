#ifndef _SCALE_FORWARD_PROPAGATION_H_
#define _SCALE_FORWARD_PROPAGATION_H_

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
#include "core/keyframes.h"

class ScaleForwardPropagation
{

private:
    std::shared_ptr<Camera> cam_;

private: // SFP parameters
    uint32_t thres_age_past_horizon_; // SFP parameter
    uint32_t thres_age_use_;
    uint32_t thres_age_recon_;
    float thres_parallax_use_;
    float thres_parallax_recon_;
    
    float thres_flow_;

public:
    ScaleForwardPropagation(const std::shared_ptr<Camera>& cam);
    ~ScaleForwardPropagation();

  void runSFP(
        const LandmarkPtrVec& lmvec, const FramePtrVec& framevec, const PoseSE3& dT10); // SFP module return : scale of the current motion.
 
// 외부에서 입력하는 파라미터.
public:
    // Turning region detection parameters
    void setTurnRegion_ThresPsi(float psi);
    void setTurnRegion_ThresCountTurn(uint32_t thres_cnt_turn);

    // Scale Forward Propagation parameters
    void setSFP_ThresAgePastHorizon(uint32_t age_past_horizon);
    void setSFP_ThresAgeUse(uint32_t age_use);
    void setSFP_ThresAgeRecon(uint32_t age_recon);
    void setSFP_ThresParallaxUse(float thres_parallax_use);
    void setSFP_ThresParallaxRecon(float thres_parallax_recon);

    // Absolute Scale Recovery parameters

// Functions related to the Scale Forward Propagation 
private:
    void solveLeastSquares_SFP(const SpMat& AtA, const SpVec& Atb, uint32_t M_tmp,
        SpVec& theta);

    void calcAinvVec_SFP(const SpMat& AA, std::vector<Mat33>& Ainv_vec, uint32_t M_tmp);

    void calcAinvB_SFP(const std::vector<Mat33>& Ainv_vec, const SpVec& B, uint32_t M_tmp,
        SpVec& AinvB);


};

#endif