#ifndef _SPARSE_BA_PARAMETERS_H_
#define _SPARSE_BA_PARAMETERS_H_
#include <iostream>
#include <vector>
#include <map>
#include <unordered_map>
#include <set>
#include <unordered_set>

#include "eigen3/Eigen/Dense"

#include "core/defines/define_macro.h"
#include "core/defines/define_type.h"

#include "core/visual_odometry/landmark.h"
#include "core/visual_odometry/frame.h"

#include "core/visual_odometry/ba_solver/define_ba_type.h"
#include "core/visual_odometry/ba_solver/landmark_ba.h"

#include "core/util/geometry_library.h"

/// @brief parameter class for Sparse Bundle Adjustment
class SparseBAParameters
{
  // Stereo related.
private:
  bool is_stereo_mode_;
  _BA_PoseSE3 T_stereo_;     // Stereo pose from the left to right (the first to the second)
  _BA_PoseSE3 T_stereo_inv_; // Stereo pose from the left to right (the first to the second)

private: // Reference Pose and scaling factor for numerical stability
  _BA_PoseSE3 Twj_ref_;
  _BA_PoseSE3 Tjw_ref_;
  _BA_Numeric pose_scale_;
  _BA_Numeric inv_pose_scale_;

private: // all frames and landmarks used for BA.
  std::unordered_set<FramePtr> frameset_all_;
  std::unordered_set<LandmarkPtr> landmarkset_all_;

private:
  int N_;        // total number of frames
  int N_opt_;    // the number of optimizable frames
  int N_nonopt_; // the number of non-optimizable frames (to prevent gauge freedom)

  /*
      Example 1) monocular mode

          N_        = 28
          N_opt_    = 20
          N_nonopt_ = 8

          N_ = N_opt_ + N_nonopt_ should hold.


      Example 2) stereo mode

          N_        = 28 (indeed 14 stereo frames)
          N_opt_    = 10
          N_nonopt_ = 4

          N_ != N_opt_ + N_nonopt_ , but N_ == 2*(N_opt_ + N_nonopt_) should hold.
  */

  int M_; // total number of landmarks (all landmarks is to be optimized if they satisty the conditions.)

  int n_obs_; // the number of observations

private:
  std::vector<LandmarkBA> lmbavec_all_; // All landmarks to be optimized

  std::unordered_map<FramePtr, _BA_PoseSE3> posemap_all_; // pose map Tjw

  std::unordered_map<FramePtr, _BA_Index> indexmap_opt_; // optimization pose index map

  FramePtrVec framemap_opt_; // j-th optimization frame ptr

  // Get methods (numbers)
public:
  inline int getNumOfAllFrames() const { return N_; }
  inline int getNumOfOptimizeFrames() const { return N_opt_; }
  inline int getNumOfFixedFrames() const { return N_nonopt_; }
  inline int getNumOfOptimizeLandmarks() const { return M_; }
  inline int getNumOfObservations() const { return n_obs_; }

  // Get methods (variables)
public:
  const _BA_PoseSE3 &getPose(const FramePtr &frame) const
  {
    if (posemap_all_.find(frame) == posemap_all_.end())
      throw std::runtime_error("In 'getPose()', posemap_all_.find(frame) == posemap_all_.end()");
    return posemap_all_.at(frame);
  }

  const _BA_Index getOptPoseIndex(const FramePtr &frame) const
  {
    if (indexmap_opt_.find(frame) == indexmap_opt_.end())
      throw std::runtime_error("In 'getOptPoseIndex()', indexmap_opt_.find(frame) == indexmap_opt_.end()");
    return indexmap_opt_.at(frame);
  }

  const FramePtr &getOptFramePtr(_BA_Index j_opt) const
  {
    if (j_opt >= framemap_opt_.size())
      throw std::runtime_error("In 'getOptFramePtr()', j_opt >= framemap_opt_.size()");
    return framemap_opt_.at(j_opt);
  }

  const LandmarkBA &getLandmarkBA(_BA_Index i) const
  {
    if (i >= lmbavec_all_.size())
      throw std::runtime_error("In 'getLandmarkBA()', i >= lmbavec_all_.size()");
    return lmbavec_all_.at(i);
  }

  // Reference version
  LandmarkBA &getLandmarkBARef(_BA_Index i)
  {
    if (i >= lmbavec_all_.size())
      throw std::runtime_error("In 'getLandmarkBARef()', i >= lmbavec_all_.size()");
    return lmbavec_all_.at(i);
  }

  const std::unordered_set<FramePtr> &getAllFrameset() const
  {
    return frameset_all_;
  }

  const std::unordered_set<LandmarkPtr> &getAllLandmarkset() const
  {
    return landmarkset_all_;
  }

  const std::unordered_map<FramePtr, _BA_PoseSE3> &getPosemap() const
  {
    return posemap_all_;
  }

  const _BA_PoseSE3 &getStereoPose() const
  {
    return T_stereo_;
  }

  // Update and get methods (Pose and Point)
public:
  void updateOptPoint(_BA_Index i, const _BA_Point &X_update)
  {
    if (i >= M_ || i < 0)
      throw std::runtime_error("In 'updateOptPoint()', i >= M_ || i < 0");
    lmbavec_all_[i].X(0) = X_update(0);
    lmbavec_all_[i].X(1) = X_update(1);
    lmbavec_all_[i].X(2) = X_update(2);
  }

  void updateOptPose(_BA_Index j_opt, const _BA_PoseSE3 &Tjw_update)
  {
    if (j_opt >= N_opt_ || j_opt < 0)
      throw std::runtime_error("In 'updateOptPose()', j_opt >= N_opt_  || j_opt < 0");

    const FramePtr &kf_opt = framemap_opt_.at(j_opt);
    posemap_all_.at(kf_opt) = Tjw_update;

    if (std::isnan(Tjw_update.norm()))
      throw std::runtime_error("In 'updateOptPose()', Tjw update nan!");
  }

  const _BA_Point &getOptPoint(_BA_Index i)
  {
    if (i >= M_ || i < 0)
      throw std::runtime_error("In 'getOptPoint()', i >= M_ || i < 0");
    return lmbavec_all_[i].X;
  }

  const LandmarkPtr &getOptLandmarkPtr(_BA_Index i)
  {
    if (i >= M_ || i < 0)
      throw std::runtime_error("In 'getOptLandmarkPtr()', i >= M_ || i < 0");
    return lmbavec_all_[i].lm;
  }

  LandmarkPtr &getOptLandmarkPtrRef(_BA_Index i)
  {
    if (i >= M_ || i < 0)
      throw std::runtime_error("In 'getOptLandmarkPtrRef()', i >= M_ || i < 0");
    return lmbavec_all_[i].lm;
  }

  const _BA_PoseSE3 &getOptPose(_BA_Index j_opt)
  {
    if (j_opt >= N_opt_ || j_opt < 0)
      throw std::runtime_error("In 'getOptPose()', j_opt >= N_opt_ || j_opt < 0");
    const FramePtr &kf_opt = framemap_opt_.at(j_opt);
    return posemap_all_.at(kf_opt);
  }

  // Find methods
public:
  const bool isOptFrame(const FramePtr &f) const { return (indexmap_opt_.find(f) != indexmap_opt_.end()); }
  const bool isFixFrame(const FramePtr &f) const { return (indexmap_opt_.find(f) == indexmap_opt_.end()); }
  const bool isStereoMode() const { return is_stereo_mode_; }

public:
  _BA_Point warpToRef(const _BA_Point &X)
  {
    _BA_Point Xw = Tjw_ref_.block<3, 3>(0, 0) * X + Tjw_ref_.block<3, 1>(0, 3);
    return Xw;
  }

  _BA_Point warpToWorld(const _BA_Point &X)
  {
    _BA_Point Xw = Twj_ref_.block<3, 3>(0, 0) * X + Twj_ref_.block<3, 1>(0, 3);
    return Xw;
  }

  _BA_PoseSE3 changeInvPoseWorldToRef(const _BA_PoseSE3 &Tjw)
  {
    _BA_PoseSE3 Tjref = Tjw * Twj_ref_;
    return Tjref;
  }

  _BA_PoseSE3 changeInvPoseRefToWorld(const _BA_PoseSE3 &Tjref)
  {
    _BA_PoseSE3 Tjw = Tjref * Tjw_ref_;
    return Tjw;
  }

  _BA_Point scalingPoint(const _BA_Point &X)
  {
    _BA_Point X_scaled = X * inv_pose_scale_;
    return X_scaled;
  }

  _BA_Point recoverOriginalScalePoint(const _BA_Point &X)
  {
    _BA_Point X_recovered = X * pose_scale_;
    return X_recovered;
  }

  _BA_PoseSE3 scalingPose(const _BA_PoseSE3 &Tjw)
  {
    _BA_PoseSE3 Tjw_scaled = Tjw;
    Tjw_scaled(0, 3) *= inv_pose_scale_;
    Tjw_scaled(1, 3) *= inv_pose_scale_;
    Tjw_scaled(2, 3) *= inv_pose_scale_;
    return Tjw_scaled;
  }

  _BA_PoseSE3 recoverOriginalScalePose(const _BA_PoseSE3 &Tjw_scaled)
  {
    _BA_PoseSE3 Tjw_org = Tjw_scaled;
    Tjw_org(0, 3) *= pose_scale_;
    Tjw_org(1, 3) *= pose_scale_;
    Tjw_org(2, 3) *= pose_scale_;
    return Tjw_org;
  }

  // Set methods
public:
  SparseBAParameters()
      : N_(0), N_opt_(0), N_nonopt_(0), M_(0), n_obs_(0),
        pose_scale_(10.0), inv_pose_scale_(1.0 / pose_scale_), is_stereo_mode_(false)
  {

    T_stereo_.setIdentity();
    T_stereo_inv_ = geometry::inverseSE3(T_stereo_);
    std::cerr << "SparseBAParameters is in 'monocular' mode.\n";
  }

  SparseBAParameters(bool is_stereo, const PoseSE3 &T_stereo)
      : N_(0), N_opt_(0), N_nonopt_(0), M_(0), n_obs_(0),
        pose_scale_(10.0), inv_pose_scale_(1.0 / pose_scale_), is_stereo_mode_(is_stereo)
  {
    if (!is_stereo_mode_)
      throw std::runtime_error("'is_stereo_mode_' should be set to 'true' when T_stereo is given!");

    std::cerr << "SparseBAParameters is in 'stereo' mode.\n";

    T_stereo_ << T_stereo(0, 0), T_stereo(0, 1), T_stereo(0, 2), T_stereo(0, 3),
        T_stereo(1, 0), T_stereo(1, 1), T_stereo(1, 2), T_stereo(1, 3),
        T_stereo(2, 0), T_stereo(2, 1), T_stereo(2, 2), T_stereo(2, 3),
        0, 0, 0, 1;

    T_stereo_inv_ = geometry::inverseSE3(T_stereo_);
  }

  ~SparseBAParameters()
  {
    std::cerr << "Sparse BA Parameters is deleted.\n";
  }

  void setPosesAndPoints(
      const FramePtrVec &frames,
      const _BA_IndexVec &idx_fix,
      const _BA_IndexVec &idx_optimize)
  {
    /*
        - Note -
        right image는 observation을 제공하지만, 해당 이미지의 포즈는 최적화 대상이 아님.
        right image와 관련된 left image 의 pose만 최적화 함.

        '모든 프레임 셋'에 left/right images 모두 포함.
        '
    */

    if (is_stereo_mode_)
    {
      T_stereo_ = scalingPose(T_stereo_);
      T_stereo_inv_ = scalingPose(T_stereo_inv_);
    }

    // Threshold for landmark usage
    constexpr int THRES_MINIMUM_SEEN = 2; // landmark should be seen on at least two stereo frames.

    N_ = static_cast<int>(frames.size());
    N_nonopt_ = static_cast<int>(idx_fix.size());
    N_opt_ = static_cast<int>(idx_optimize.size());

    std::cerr << "In 'SparseBAParameters::setPosesAndPoints()', N: " << N_ << ", N_fix: " << N_nonopt_ << ", N_opt: " << N_opt_ << std::endl;
    std::cerr << "stereo mode? : " << is_stereo_mode_ << std::endl;
    if (is_stereo_mode_)
    {
      if (N_ != (N_nonopt_ + N_opt_) * 2)
        throw std::runtime_error("In 'SparseBAParameters::setPosesAndPoints()', stereo mode is set, but N != 2*N_fix + 2*N_opt ");
    }
    else
    {
      if (N_ != (N_nonopt_ + N_opt_))
        throw std::runtime_error("In 'SparseBAParameters::setPosesAndPoints()', monocular mode is set, but N != N_fix + N_opt ");
    }

    // 1) get all window keyframes
    std::unordered_set<LandmarkPtr> lmset_window; // 안겹치는 랜드마크들
    std::unordered_set<FramePtr> frameset_window; // 윈도우 내부 키프레임들
    FramePtrVec kfvec_window;                     // 윈도우 내부의 키프레임들
    for (const auto &f : frames)
    {
      // 모든 keyframe in window 순회
      kfvec_window.push_back(f); // window keyframes 저장.
      frameset_window.insert(f); // left and right 모두 포함.

      for (const auto &lm : f->getRelatedLandmarkPtr())
      {
        // 현재 keyframe에서 보인 모든 landmark 순회
        if (lm->isTriangulated() && lm->isAlive()) // age > THRES, triangulate() == true 경우 포함.
          lmset_window.insert(lm);
      }
    }
    std::cerr << "In 'setPosesAndPoints()', SIZE kfvec_window : " << kfvec_window.size() << std::endl;
    std::cerr << "In 'setPosesAndPoints()', SIZE frameset_window : " << frameset_window.size() << std::endl;

    // 1-1) get reference pose.
    const PoseSE3 &Twj_ref_float = kfvec_window.front()->getPose(); // 맨 첫 자세.
    Twj_ref_ << Twj_ref_float(0, 0), Twj_ref_float(0, 1), Twj_ref_float(0, 2), Twj_ref_float(0, 3),
        Twj_ref_float(1, 0), Twj_ref_float(1, 1), Twj_ref_float(1, 2), Twj_ref_float(1, 3),
        Twj_ref_float(2, 0), Twj_ref_float(2, 1), Twj_ref_float(2, 2), Twj_ref_float(2, 3),
        0.0, 0.0, 0.0, 1.0;

    Tjw_ref_ = geometry::inverseSE3(Twj_ref_);

    // 2) make LandmarkBAVec
    for (const auto &lm : lmset_window)
    {
      // keyframe window 내에서 보였던 모든 landmark를 순회.
      LandmarkBA lm_ba;
      lm_ba.lm = lm; // landmark pointer

      // warp to Reference frame.
      const Point &X_float = lm->get3DPoint();
      lm_ba.X << X_float(0), X_float(1), X_float(2); // 3D point represented in the global frame.

      lm_ba.X = warpToRef(lm_ba.X);
      lm_ba.X = scalingPoint(lm_ba.X);

      lm_ba.kfs_seen.reserve(300);
      lm_ba.pts_on_kfs.reserve(300);
      lm_ba.err_on_kfs.reserve(300);

      // 현재 landmark가 보였던 keyframes을 저장한다.
      for (size_t j = 0; j < static_cast<int>(lm->getRelatedKeyframePtr().size()); ++j)
      {
        const FramePtr &kf = lm->getRelatedKeyframePtr()[j]; // left 든 right든 상관 없음.
        const Pixel &pt = lm->getObservationsOnKeyframes()[j];
        if (frameset_window.find(kf) != frameset_window.end()) // 윈도우 내부에 있는 프레임이면 추가.
        {
          // window keyframe만으로 제한
          lm_ba.kfs_seen.push_back(kf);
          lm_ba.pts_on_kfs.emplace_back(pt.x, pt.y);
          lm_ba.err_on_kfs.push_back(0.0);
        }
      }

      // 충분히 많은 keyframes in window에서 보인 landmark만 최적화에 포함.
      if (lm_ba.kfs_seen.size() >= THRES_MINIMUM_SEEN)
      {
        lmbavec_all_.push_back(lm_ba);
        landmarkset_all_.insert(lm);

        for (int j = 0; j < static_cast<int>(lm_ba.kfs_seen.size()); ++j) // all related keyframes.
          frameset_all_.insert(lm_ba.kfs_seen[j]);
      }
    }

    // 3) re-initialize N, N_fix, M
    N_ = static_cast<int>(frameset_all_.size());
    if (is_stereo_mode_)
      N_nonopt_ = (N_ - 2 * N_opt_) / 2;
    else
      N_nonopt_ = N_ - N_opt_;

    M_ = static_cast<int>(lmbavec_all_.size());
    std::cerr << "Recomputed N: " << N_ << ", N_fix + N_opt: " << N_nonopt_ << "+" << N_opt_ << std::endl;

    // 4) set poses for all frames
    for (const auto &f : frameset_all_)
    {
      if (!f->isRightImage()) // Left image only.
      {
        const PoseSE3 &Tjw_float = f->getPoseInv();
        _BA_PoseSE3 Tjw_tmp;
        Tjw_tmp << Tjw_float(0, 0), Tjw_float(0, 1), Tjw_float(0, 2), Tjw_float(0, 3),
            Tjw_float(1, 0), Tjw_float(1, 1), Tjw_float(1, 2), Tjw_float(1, 3),
            Tjw_float(2, 0), Tjw_float(2, 1), Tjw_float(2, 2), Tjw_float(2, 3),
            0.0, 0.0, 0.0, 1.0;

        Tjw_tmp = changeInvPoseWorldToRef(Tjw_tmp);
        Tjw_tmp = scalingPose(Tjw_tmp);

        posemap_all_.insert(std::pair<FramePtr, _BA_PoseSE3>(f, Tjw_tmp));
      }
    }

    // 5) set optimizable keyframes (posemap, indexmap, framemap)
    _BA_Index cnt_idx = 0;
    for (size_t jj = 0; jj < static_cast<int>(idx_optimize.size()); ++jj)
    {
      const int j = idx_optimize.at(jj);
      if (!frames[j]->isRightImage()) // Left image only.
      {
        indexmap_opt_.insert({frames[j], cnt_idx});
        framemap_opt_.push_back(frames[j]);
        ++cnt_idx;
      }
    }

    // check
    if (cnt_idx != indexmap_opt_.size())
      throw std::runtime_error("cnt_idx != indexmap_opt_.size()");

    // 6) set optimization values
    n_obs_ = 0; // the number of total observations (2*n_obs == len_residual)
    for (const auto &lm_ba : lmbavec_all_)
      n_obs_ += static_cast<int>(lm_ba.kfs_seen.size()); // residual 크기.

    const int len_residual = 2 * n_obs_;
    const int len_parameter = 6 * N_opt_ + 3 * M_;
    printf("| Bundle Adjustment Statistics:\n");
    printf("|  -        # of total images: %d images \n", N_);
    printf("|  -           -  opt. images: %d images \n", N_opt_);
    printf("|  -           -  fix  images: %d images \n", N_nonopt_);
    printf("|  -        # of opti. points: %d landmarks \n", M_);
    printf("|  -        # of observations: %d \n", n_obs_);
    printf("|  -            Jacobian size: %d rows x %d cols\n", len_residual, len_parameter);
    printf("|  -            Residual size: %d rows\n\n", len_residual);
  };
};

#endif