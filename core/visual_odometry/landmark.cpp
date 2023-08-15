#include "core/visual_odometry/landmark.h"

PixelVec Landmark::patt_ = PixelVec();

Landmark::Landmark()
    : id_(landmark_counter_++), age_(0),
      Xw_(0, 0, 0),
      is_alive_(true),
      is_tracked_(true),
      is_triangulated_(false),
      is_bundled_(false)
{
  // Reserve storages
  observations_.reserve(50);
  related_frames_.reserve(50);

  observations_on_keyframes_.reserve(20);
  view_sizes_.reserve(20);
  related_keyframes_.reserve(20);

  // Initialize parallax and opt flow.
  min_parallax_ = 1000.0f;
  max_parallax_ = 0.0f;
  avg_parallax_ = 0.0f;
  last_parallax_ = 0.0f;
}

Landmark::Landmark(const Pixel &p, const FramePtr &frame)
    : id_(landmark_counter_++), age_(0),
      Xw_(0, 0, 0),
      is_alive_(true),
      is_tracked_(true),
      is_triangulated_(false),
      is_bundled_(false)
{
  // Reserve storages
  observations_.reserve(50);
  related_frames_.reserve(50);

  observations_on_keyframes_.reserve(20);
  view_sizes_.reserve(20);
  related_keyframes_.reserve(20);

  // Initialize parallax and opt flow.
  min_parallax_ = 1000.0f;
  max_parallax_ = 0.0f;
  avg_parallax_ = 0.0f;
  last_parallax_ = 0.0f;

  // Add observation
  addObservationAndRelatedFrame(p, frame);
}

Landmark::~Landmark()
{
  // std::cout << "Landmark destructor called, ID [" << id_ << "]\n";
}

void Landmark::set3DPoint(const Point &Xw)
{
  Xw_ = Xw;
  is_triangulated_ = true;
}

void Landmark::setUntracked()
{
  is_tracked_ = false;
}

void Landmark::setBundled()
{
  is_triangulated_ = true;
  is_bundled_ = true;
}

void Landmark::addObservationAndRelatedFrame(const Pixel &p, FrameConstPtr &frame)
{
  CameraConstPtr &cam = frame->getCamera();

  // push observation.
  if (!frame->isRightImage())
    ++age_;

  observations_.push_back(p);
  related_frames_.push_back(frame);

  /* At the first image where this landmark is firstly observed,
   we save the image patch around the pixel point.
  */
  if (observations_.size() == 1)
  {
    // Set patch
    const float ax = p.x - floor(p.x);
    const float ay = p.y - floor(p.y);
    const float axay = ax * ay;

    PixelVec pts_patt(patt_.size());
    for (size_t i = 0; i < patt_.size(); ++i)
      pts_patt[i] = p + patt_[i];

    if (ax < 0 || ax > 1 || ay < 0 || ay > 1)
      throw std::runtime_error("ax ay invalid!");

    return;
  }

  // Calculate 'parallax' w.r.t. the oldest pixel
  const Pixel &p0 = observations_.front();
  const Pixel &p1 = observations_.back();

  PoseSE3 T01 = related_frames_.front()->getPoseInv() * related_frames_.back()->getPose();

  Point x0, x1;
  x0 << (p0.x - cam->cx()) * cam->fxinv(), (p0.y - cam->cy()) * cam->fyinv(), 1.0f;
  x1 << (p1.x - cam->cx()) * cam->fxinv(), (p1.y - cam->cy()) * cam->fyinv(), 1.0f;
  x1 = T01.block<3, 3>(0, 0) * x1;

  float costheta = x0.dot(x1) / (x0.norm() * x1.norm());
  if (costheta >= 1.0f)
    costheta = 0.99999f;
  if (costheta <= -1.0f)
    costheta = -0.99999f;

  const float parallax_curr = acosf(costheta);
  last_parallax_ = parallax_curr;
  if (max_parallax_ <= parallax_curr)
    max_parallax_ = parallax_curr;
  if (min_parallax_ >= parallax_curr)
    min_parallax_ = parallax_curr;

  const float invage = 1.0f / static_cast<float>(age_);
  avg_parallax_ = avg_parallax_ * static_cast<float>(age_ - 1);
  avg_parallax_ += parallax_curr;
  avg_parallax_ *= invage;
}

void Landmark::addObservationAndRelatedKeyframe(const Pixel &p, const FramePtr &kf)
{
  observations_on_keyframes_.push_back(p);
  related_keyframes_.push_back(kf);
}

void Landmark::changeLastObservation(const Pixel &p)
{
  observations_.back() = p;
}

void Landmark::setDead()
{
  is_alive_ = false;
  is_tracked_ = false;
}

int Landmark::getID() const { return id_; }
int Landmark::getAge() const { return age_; }
const Point &Landmark::get3DPoint() const { return Xw_; }
const PixelVec &Landmark::getObservations() const { return observations_; }
const FramePtrVec &Landmark::getRelatedFramePtr() const { return related_frames_; }
const PixelVec &Landmark::getObservationsOnKeyframes() const { return observations_on_keyframes_; }
const FramePtrVec &Landmark::getRelatedKeyframePtr() const { return related_keyframes_; }

// const std::vector<float>& Landmark::getImagePatchVec() const { return I0_patt_; }
// const std::vector<float>& Landmark::getDuPatchVec() const    { return du0_patt_; }
// const std::vector<float>& Landmark::getDvPatchVec() const    { return dv0_patt_; }
// const MaskVec&            Landmark::getMaskPatchVec()  const { return mask_patt_; }

const bool &Landmark::isAlive() const { return is_alive_; }
const bool &Landmark::isTracked() const { return is_tracked_; }
const bool &Landmark::isTriangulated() const { return is_triangulated_; }
const bool &Landmark::isBundled() const { return is_bundled_; }

float Landmark::getMinParallax() const { return min_parallax_; }
float Landmark::getMaxParallax() const { return max_parallax_; }
float Landmark::getAvgParallax() const { return avg_parallax_; }
float Landmark::getLastParallax() const { return last_parallax_; }

/*
====================================================================================
====================================================================================
================================  LandmarkTracking  ================================
====================================================================================
====================================================================================
*/

LandmarkTracking::LandmarkTracking()
{
  pts0.reserve(1000);
  pts1.reserve(1000);
  lms.reserve(1000);
  scale_change.reserve(1000);
  n_pts = 0;
}

LandmarkTracking::LandmarkTracking(const LandmarkTracking &lmtrack, const MaskVec &mask)
{
  if (lmtrack.pts0.size() != lmtrack.pts1.size() || lmtrack.pts0.size() != lmtrack.lms.size() || lmtrack.pts0.size() != mask.size())
    throw std::runtime_error("lmtrack.pts0.size() != lmtrack.pts1.size() || lmtrack.pts0.size() != lmtrack.lms.size() || lmtrack.pts0.size() != mask.size()");

  int n_pts_input = lmtrack.pts0.size();

  std::vector<int> index_valid;
  index_valid.reserve(n_pts_input);
  int cnt_alive = 0;
  for (int i = 0; i < n_pts_input; ++i)
  {
    if (mask[i] && lmtrack.lms[i]->isAlive() && lmtrack.lms[i]->isTracked())
    {
      index_valid.push_back(i);
      ++cnt_alive;
    }
    else
      lmtrack.lms[i]->setUntracked();
  }

  // set
  n_pts = cnt_alive;

  pts0.resize(n_pts);
  pts1.resize(n_pts);
  lms.resize(n_pts);
  scale_change.resize(n_pts);
  for (int i = 0; i < cnt_alive; ++i)
  {
    const int &idx = index_valid[i];

    pts0[i] = lmtrack.pts0[idx];
    pts1[i] = lmtrack.pts1[idx];
    scale_change[i] = lmtrack.scale_change[idx];
    lms[i] = lmtrack.lms[idx];
  }
}

LandmarkTracking::LandmarkTracking(const PixelVec &pts0_in, const PixelVec &pts1_in, const LandmarkPtrVec &lms_in)
{
  if (pts0_in.size() != pts1_in.size() || pts0_in.size() != lms_in.size())
    throw std::runtime_error("pts0_in.size() != pts1_in.size() || pts0_in.size() != lms_in.size()");

  int n_pts_input = pts0_in.size();

  std::vector<int> index_valid;
  index_valid.reserve(n_pts_input);
  int cnt_alive = 0;
  for (int i = 0; i < n_pts_input; ++i)
  {
    if (lms_in[i]->isAlive())
    {
      index_valid.push_back(i);
      ++cnt_alive;
    }
    else
      continue;
  }

  // set
  n_pts = cnt_alive;

  pts0.resize(n_pts);
  pts1.resize(n_pts);
  lms.resize(n_pts);
  scale_change.resize(n_pts);
  for (int i = 0; i < cnt_alive; ++i)
  {
    const int &idx = index_valid[i];

    pts0[i] = pts0_in[idx];
    pts1[i] = pts1_in[idx];
    scale_change[i] = 0;
    lms[i] = lms_in[idx];
  }
}

/*
====================================================================================
====================================================================================
=============================  StereoLandmarkTracking  =============================
====================================================================================
====================================================================================
*/
StereoLandmarkTracking::StereoLandmarkTracking()
{
  pts_l0.reserve(1000);
  pts_l1.reserve(1000);
  pts_r0.reserve(1000);
  pts_r1.reserve(1000);

  lms.reserve(1000);

  n_pts = 0;
}

StereoLandmarkTracking::StereoLandmarkTracking(const StereoLandmarkTracking &slmtrack, const MaskVec &mask)
{

  if (slmtrack.pts_l0.size() != slmtrack.pts_l1.size() || slmtrack.pts_l0.size() != slmtrack.pts_r0.size() || slmtrack.pts_l0.size() != slmtrack.pts_r1.size() || slmtrack.pts_l0.size() != slmtrack.lms.size() || slmtrack.pts_l0.size() != mask.size())
    throw std::runtime_error("slmtrack.pts_left_0.size() != slmtrack.pts_left_1.size() || slmtrack.pts_left_0.size() != slmtrack.pts_right_0.size() || slmtrack.pts_left_0.size() != slmtrack.pts_right_1.size() || slmtrack.pts_left_0.size() != slmtrack.lms.size() || slmtrack.pts_left_0.size() != mask.size()");

  int n_pts_input = slmtrack.pts_l0.size();

  std::vector<int> index_valid;
  index_valid.reserve(n_pts_input);
  int cnt_alive = 0;
  for (int i = 0; i < n_pts_input; ++i)
  {
    if (mask[i] && slmtrack.lms[i]->isAlive() && slmtrack.lms[i]->isTracked())
    {
      index_valid.push_back(i);
      ++cnt_alive;
    }
    else
      slmtrack.lms[i]->setUntracked();
  }

  // set
  n_pts = cnt_alive;

  pts_l0.resize(n_pts);
  pts_l1.resize(n_pts);
  pts_r0.resize(n_pts);
  pts_r1.resize(n_pts);
  lms.resize(n_pts);

  for (int i = 0; i < cnt_alive; ++i)
  {
    const int &idx = index_valid[i];

    pts_l0[i] = slmtrack.pts_l0[idx];
    pts_l1[i] = slmtrack.pts_l1[idx];
    pts_r0[i] = slmtrack.pts_r0[idx];
    pts_r1[i] = slmtrack.pts_r1[idx];
    lms[i] = slmtrack.lms[idx];
  }
}