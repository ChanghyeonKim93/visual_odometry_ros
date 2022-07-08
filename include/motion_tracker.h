#ifndef _MOTION_TRACKER_H_
#define _MOTION_TRACKER_H_

#include <iostream>
#include <exception>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// #include "frame.h"

class MotionTracker{
public:
    MotionTracker();
    ~MotionTracker();

    cv::Mat trackCurrentImage(const cv::Mat& img, const double& timestamp);

private:
    void track();
    void monocularInitialization();

public:

    enum TrackingState{
        SYSTEM_NOT_READY=-1,
        NO_IMAGES_YET=0,
        NOT_INITIALIZED=1,
        OK=2,
        LOST=3
    };


private:

    cv::Mat img_current_;
    // Frame* frame_current_;

    TrackingState track_state_;

};

#endif