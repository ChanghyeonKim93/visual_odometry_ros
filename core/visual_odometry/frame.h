#ifndef _FRAME_H_
#define _FRAME_H_

#include <iostream>
#include <vector>
#include <memory>

#include "eigen3/Eigen/Dense"
#include "opencv4/opencv2/core.hpp"

#include "core/defines/define_type.h"

#include "core/visual_odometry/landmark.h"
#include "core/visual_odometry/camera.h"

#include "core/util/geometry_library.h"

class Frame;
class StereoFrame;

/// @brief Image frame class
class Frame
{
private:
  std::shared_ptr<Camera> cam_; // camera class pointer

private:
  int id_;           // unique ID of this frame.
  double timestamp_; // timestamp of this frame in second.

  // Poses of the frame
  PoseSE3 Twc_; // SE(3) w.r.t. the global frame {W}.
  PoseSE3 Tcw_; // Inverse of Twc_.

  PoseSE3 dT10_;
  PoseSE3 dT01_;

  // Feature related storages
  LandmarkPtrVec related_landmarks_;
  PixelVec pts_seen_;

  // Flags
  bool is_keyframe_;
  bool is_keyframe_in_window_;
  bool is_poseonlyBA_success_; //

  // For stereo (right) frame
private:
  bool is_right_image_; // stereo image.
  FramePtr frame_left_; // Frame pointer to 'left frame' of this right frame. In this project, we consider the 'left image' is the reference image of the stereo frames. All poses of the right frames are represented by their 'left frames'.

public:                                 // static counter.
  inline static int frame_counter_ = 0; // Unique frame counter. (static)
  inline static int max_pyr_lvl_ = 5;

public:
  Frame(const std::shared_ptr<Camera> &cam, bool is_right_image = false, FrameConstPtr &frame_left = nullptr);
  Frame(const std::shared_ptr<Camera> &cam, const double &timestamp,
        bool is_right_image = false, FrameConstPtr &frame_left = nullptr);

  // Set
  void setPose(const PoseSE3 &Twc);
  void setPoseDiff10(const PoseSE3 &dT10);
  void setTimestamp(const double &timestamp);

  void setPtsSeenAndRelatedLandmarks(const PixelVec &pts, const LandmarkPtrVec &landmarks);

  void makeThisKeyframe();
  void outOfKeyframeWindow();

  // Get
  const int &getID() const;
  const PoseSE3 &getPose() const;
  const PoseSE3 &getPoseInv() const;
  const PoseSE3 &getPoseDiff10() const;
  const PoseSE3 &getPoseDiff01() const;
  CameraConstPtr &getCamera() const;
  const LandmarkPtrVec &getRelatedLandmarkPtr() const;
  const PixelVec &getPtsSeen() const;
  const double &getTimestamp() const;
  bool isKeyframe() const;
  bool isKeyframeInWindow() const;

  // For stereo frame
public:
  bool isRightImage() const;
  FrameConstPtr &getLeftFramePtr() const;

  // For motion update flags
public:
  bool isPoseOnlySuccess() const;
  void setPoseOnlyFailed();
};

/// @brief Stereo frame structure. (not used now.)
class StereoFrame
{
private:
  FramePtr left_;  // frame pointr to lower frame (same as left frame)
  FramePtr right_; // frame pointr to upper frame (same as right frame)

public:
  StereoFrame(FrameConstPtr &frame_left, FrameConstPtr &frame_right);
  StereoFrame(CameraConstPtr &cam_left, CameraConstPtr &cam_right, double timestamp);

public:
  FrameConstPtr &getLeft() const;
  FrameConstPtr &getRight() const;

public:
  void setStereoPoseByLeft(const PoseSE3 &Twc_left, const PoseSE3 &T_lr);
  void setStereoPtsSeenAndRelatedLandmarks(const PixelVec &pts_l1, const PixelVec &pts_r1, const LandmarkPtrVec &lms);
};

#endif