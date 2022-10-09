#ifndef _SCALE_ESTIMATOR_H_
#define _SCALE_ESTIMATOR_H_

#include <iostream>
#include <string>
#include <exception>
#include <numeric>
#include <vector>

// Related to the multithreading.
#include <thread>
#include <mutex>
#include <chrono>
#include <future>
#include <condition_variable>

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

/// @brief ScaleEstimator class. This class runs on another thread.
class ScaleEstimator
{

// 동작방식 : 따로 thread가 돌며, scale update가 수행되어야 하는 상황에 flag와 함께 frame정보를 넘겨준다.
// ScaleForwardPropagation : Keyframes in window, current pose, previous poses
// recoverAbsoluteScalesBetweenTurns (ASR): 
// Input: frames_all, idx_t1, idx_t2, camera
//        내부에서는 local bundle adjustment 처럼 구동한다.

public:
    ScaleEstimator(
        const std::shared_ptr<Camera> cam,
        const std::shared_ptr<std::mutex> mut, 
        const std::shared_ptr<std::condition_variable> cond_var,
        const std::shared_ptr<bool> flag_do_ASR);
    ~ScaleEstimator();

// Two modules: Scale Forward Propagation (SFP), Absolute Scale Recovery (ASR)
// SFP: called at every new keyframes
// ASR: called when new turning region is detected.
private:
    void module_ScaleForwardPropagation(
        const LandmarkPtrVec& lmvec, const FramePtrVec& framevec, const PoseSE3& dT10); // SFP module return : scale of the current motion.
    void module_AbsoluteScaleRecovery(); // SFP module return : scale of the current motion.


// 이것도 class 안에서 구동 되도록 바꿀 것.
// 매 keyframe이 생성 될 때 마다 해당 keyframe을 ScaleEstimator로 전달한다. (복사 & 원본 포인터 가지고 있기)
// 
public:
    bool detectTurnRegions(const FramePtr& frame);
    bool detectTurnRegions(const std::shared_ptr<Keyframes>& keyframes, const FramePtr& frame);
    const FramePtrVec& getAllTurnRegions() const;
    const FramePtrVec& getLastTurnRegion() const;
    
    float calcSteeringAngleFromRotationMat(const Rot3& R);

// 외부에서 입력하는 파라미터.
public:
    void setTurnRegion_ThresPsi(float psi);
    void setTurnRegion_ThresCountTurn(uint32_t thres_cnt_turn);

    void setSFP_ThresAgePastHorizon(uint32_t age_past_horizon);
    void setSFP_ThresAgeUse(uint32_t age_use);
    void setSFP_ThresAgeRecon(uint32_t age_recon);
    void setSFP_ThresParallaxUse(float thres_parallax_use);
    void setSFP_ThresParallaxRecon(float thres_parallax_recon);

private:
    void runThread();
    void process(std::shared_future<void> terminate_signal);

private:
    float calcScaleByKinematics(float psi, const Pos3& u01, float L);

public:

    
// Functions related to the Scale Forward Propagation 
private:
    void solveLeastSquares_SFP(const SpMat& AtA, const SpVec& Atb, uint32_t M_tmp,
        SpVec& theta);

    void calcAinvVec_SFP(const SpMat& AA, std::vector<Mat33>& Ainv_vec, uint32_t M_tmp);

    void calcAinvB_SFP(const std::vector<Mat33>& Ainv_vec, const SpVec& B, uint32_t M_tmp,
        SpVec& AinvB);

private:
    std::shared_ptr<Camera> cam_;

private:
    std::thread thread_;
    std::shared_ptr<std::mutex> mut_;
    std::shared_ptr<std::condition_variable> cond_var_;
    std::shared_ptr<bool> flag_do_ASR_;

// Turn region parameters 
private:
    uint32_t cnt_turn_;

    uint32_t thres_cnt_turn_;
    float thres_psi_;

// SFP parameters
private:
    uint32_t thres_age_past_horizon_; // SFP parameter
    uint32_t thres_age_use_;
    uint32_t thres_age_recon_;
    float thres_parallax_use_;
    float thres_parallax_recon_;
    
    float thres_flow_;

    FramePtrVec frames_t0_;
    FramePtrVec frames_t1_;
    FramePtrVec frames_u_;

    FramePtrVec frames_all_t_;

private:

// Variables to elegantly terminate TX & RX threads
private:
    std::shared_future<void> terminate_future_;
    std::promise<void>       terminate_promise_;

};

#endif