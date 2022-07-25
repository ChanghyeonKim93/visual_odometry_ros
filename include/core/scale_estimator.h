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
#include <opencv2/core.hpp>

#include "core/type_defines.h"
#include "core/camera.h"
#include "core/mapping.h"
#include "core/landmark.h"
#include "core/frame.h"

class ScaleEstimator{
public:
    ScaleEstimator(const std::shared_ptr<std::mutex> mut, 
        const std::shared_ptr<std::condition_variable> cond_var,
        const std::shared_ptr<bool> flag_do_ASR);
    ~ScaleEstimator();

    void module_ScaleForwardPropagation(const LandmarkPtrVec& lmvec, const FramePtrVec& framevec, const PoseSE3 dT10); // SFP module return : scale of the current motion.
    void module_AbsoluteScaleRecovery(); // SFP module return : scale of the current motion.

    void detectTurnRegions(float psi, const FramePtr& frame);

private:
    void runThread();
    void process(std::shared_future<void> terminate_signal);

public:
    static std::shared_ptr<Camera> cam_;

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

    FramePtrVec frames_t0_;
    FramePtrVec frames_t1_;
    FramePtrVec frames_u_;

// Variables to elegantly terminate TX & RX threads
private:
    std::shared_future<void> terminate_future_;
    std::promise<void>       terminate_promise_;

};

#endif