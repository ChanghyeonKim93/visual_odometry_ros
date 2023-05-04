
#include "motion_estimator/motion_estimator.h"

bool MotionEstimator::poseOnlyBundleAdjustment(
    const PointVec &X, const PixelVec &pts1, const float fx, const float fy, const float cx, const float cy, const int &thres_reproj_outlier,
    Rot3 &R01_true, Pos3 &t01_true, MaskVec &mask_inlier)
{
  // X is represented in the world frame.
  if (X.size() != pts1.size())
    throw std::runtime_error("In 'poseOnlyBundleAdjustment()': X.size() != pts1.size().");

  bool is_success = true;

  size_t n_pts = X.size();
  mask_inlier.resize(n_pts);

  size_t MAX_ITER = 100;
  float THRES_HUBER = 0.5f; // pixels
  float THRES_DELTA_XI = 1e-6;
  float THRES_DELTA_ERROR = 1e-7;
  float THRES_REPROJ_ERROR = thres_reproj_outlier; // pixels

  float lambda = 0.00001f;

  const float fxinv = 1.0 / fx;
  const float fyinv = 1.0 / fy;

  PoseSE3 T01_init;
  T01_init << R01_true, t01_true, 0, 0, 0, 1;

  float err_prev = 1e10f;

  Mat66 JtWJ;
  Vec6 mJtWr;

  PoseSE3 T10_optimized = T01_init.inverse();
  for (size_t iter = 0; iter < MAX_ITER; ++iter)
  {
    mJtWr.setZero();
    JtWJ.setZero();

    const Rot3 &R10 = T10_optimized.block<3, 3>(0, 0);
    const Pos3 &t10 = T10_optimized.block<3, 1>(0, 3);

    float err_curr = 0.0f;
    float inv_npts = 1.0f / (float)n_pts;
    size_t cnt_invalid = 0;
    // Warp and project point & calculate error...
    for (size_t i = 0; i < n_pts; ++i)
    {
      const Pixel &pt = pts1[i];
      Point Xw = R10 * X[i] + t10;

      float iz = 1.0f / Xw(2);
      float xiz = Xw(0) * iz;
      float yiz = Xw(1) * iz;
      float fxxiz = fx * xiz;
      float fyyiz = fy * yiz;

      Pixel pt_warp;
      pt_warp.x = fxxiz + cx;
      pt_warp.y = fyyiz + cy;

      float rx = pt_warp.x - pt.x;
      float ry = pt_warp.y - pt.y;

      // Huber weight calculation by the Manhattan distance
      float weight = 1.0f;
      bool flag_weight = false;
      float absrxry = abs(rx) + abs(ry);
      if (absrxry >= THRES_HUBER)
      {
        weight = THRES_HUBER / absrxry;
        flag_weight = true;
      }

      if (absrxry >= THRES_REPROJ_ERROR)
      {
        mask_inlier[i] = false;
        ++cnt_invalid;
      }
      else
        mask_inlier[i] = true;

      // JtWJ, JtWr for x
      Vec6 Jt;
      Jt.setZero();
      Mat66 JtJ_tmp;
      JtJ_tmp.setZero();

      Jt(0) = fx * iz;
      Jt(1) = 0.0f;
      Jt(2) = -fxxiz * iz;
      Jt(3) = -fxxiz * yiz;
      Jt(4) = fx * (1.0f + xiz * xiz);
      Jt(5) = -fx * yiz;

      if (flag_weight)
      {
        float w_rx = weight * rx;
        float err = w_rx * rx;

        // JtWJ.noalias() += weight *(Jt*Jt.transpose());
        this->calcJtWJ_x(weight, Jt, JtJ_tmp);
        JtWJ.noalias() += JtJ_tmp;
        mJtWr.noalias() -= (w_rx)*Jt;
        err_curr += rx * rx;
      }
      else
      {
        float err = rx * rx;
        // JtWJ.noalias() += Jt*Jt.transpose();
        this->calcJtJ_x(Jt, JtJ_tmp);
        JtWJ.noalias() += JtJ_tmp;
        mJtWr.noalias() -= rx * Jt;
        err_curr += err;
      }

      // JtWJ, JtWr for y
      Jt(0) = 0.0f;
      Jt(1) = fy * iz;
      Jt(2) = -fyyiz * iz;
      Jt(3) = -fy * (1.0f + yiz * yiz);
      Jt(4) = fyyiz * xiz;
      Jt(5) = fy * xiz;

      if (flag_weight)
      {
        float w_ry = weight * ry;
        float err = w_ry * ry;
        // JtWJ.noalias()  += weight*(Jt*Jt.transpose());
        this->calcJtWJ_y(weight, Jt, JtJ_tmp);
        JtWJ.noalias() += JtJ_tmp;
        mJtWr.noalias() -= w_ry * Jt;
        err_curr += ry * ry;
      }
      else
      {
        float err = ry * ry;
        // JtWJ.noalias()  += Jt*Jt.transpose();
        this->calcJtJ_y(Jt, JtJ_tmp);
        JtWJ.noalias() += JtJ_tmp;
        mJtWr.noalias() -= ry * Jt;
        err_curr += err;
      }
    } // END FOR

    err_curr *= (inv_npts * 0.5f);
    float delta_err = abs(err_curr - err_prev);

    // Solve H^-1*Jtr;
    for (size_t i = 0; i < 6; ++i)
      JtWJ(i, i) *= (1.0f + lambda); // lambda

    PoseSE3Tangent delta_xi = JtWJ.ldlt().solve(mJtWr);

    // Update matrix
    PoseSE3 dT;
    geometry::se3Exp_f(delta_xi, dT);
    T10_optimized.noalias() = dT * T10_optimized;

    err_prev = err_curr;

    std::cout << "reproj. err. (avg): " << err_curr << ", step: " << delta_xi.transpose() << ", det: " << T10_optimized.block<3, 3>(0, 0).determinant() << std::endl;
    if (delta_xi.norm() < THRES_DELTA_XI || delta_err < THRES_DELTA_ERROR)
    {
      std::cout << "    poseonly BA stops at: " << iter << ", err: " << err_curr << ", derr: " << delta_err << ", deltaxi: " << delta_xi.norm() << ", # invalid: " << cnt_invalid << "\n";
      break;
    }
    if (iter == MAX_ITER - 1)
    {
      std::cout << "    !! WARNING !! poseonly BA stops at full iterations!!"
                << ", err: " << err_curr << ", derr: " << delta_err << ", # invalid: " << cnt_invalid << "\n";
    }
  }

  if (!std::isnan(T10_optimized.norm()))
  {
    PoseSE3 T01_update;
    T01_update << geometry::inverseSE3_f(T10_optimized);
    R01_true = T01_update.block<3, 3>(0, 0);
    t01_true = T01_update.block<3, 1>(0, 3);
  }
  else
  {
    std::cout << "!! WARNING !! poseonly BA yields NAN value!!"
              << ", T10_optimized: \n"
              << T10_optimized << "\n";
    is_success = false; // if nan, do not update.
  }

  return is_success;
};

bool MotionEstimator::poseOnlyBundleAdjustment_Stereo(
    const PointVec &X, const PixelVec &pts_l1, const PixelVec &pts_r1,
    const float fx_l, const float fy_l, const float cx_l, const float cy_l, const float fx_r, const float fy_r, const float cx_r, const float cy_r,
    const PoseSE3 &T_lr, float thres_reproj_outlier,
    PoseSE3 &T01, MaskVec &mask_inlier)
{
  if (!is_stereo_mode_)
    throw std::runtime_error("In 'poseOnlyBundleAdjustment_Stereo()', is_stereo_mode_ == false");

  PoseSE3 T_rl = geometry::inverseSE3_f(T_lr);

  // X is represented in the reference frame.
  if (X.size() != pts_l1.size() || X.size() != pts_r1.size())
    throw std::runtime_error("In 'poseOnlyStereoBundleAdjustment()': X.size() != pts_l1.size() || X.size() != pts_r1.size().");

  bool is_success = true;

  size_t n_pts = X.size();
  mask_inlier.assign(n_pts, true);

  size_t MAX_ITER = 100;
  float THRES_HUBER = 0.5f; // pixels
  float THRES_DELTA_XI = 1e-6;
  float THRES_DELTA_ERROR = 1e-7;
  float THRES_REPROJ_ERROR = thres_reproj_outlier; // pixels

  float lambda = 0.00001f;

  const float fx_l_inv = 1.0/fx_l;
  const float fy_l_inv = 1.0/fy_l;

  const float fx_r_inv = 1.0/fx_r;
  const float fy_r_inv = 1.0/fy_r;

  float err_prev = 1e10f;
  PoseSE3 T10_optimized;
  T10_optimized << T01.block<3, 3>(0, 0).transpose(), -T01.block<3, 3>(0, 0).transpose() * T01.block<3, 1>(0, 3), 0, 0, 0, 1;

  Mat66 JtWJ;
  Vec6 mJtWr;
  for (size_t iter = 0; iter < MAX_ITER; ++iter)
  {
    JtWJ.setZero();
    mJtWr.setZero();

    const Rot3 &R10 = T10_optimized.block<3, 3>(0, 0);
    const Pos3 &t10 = T10_optimized.block<3, 1>(0, 3);

    float err_curr = 0.0f;
    float inv_npts = 1.0f / (float)n_pts;
    size_t cnt_invalid = 0;
    // Warp and project point & calculate error...
    for (size_t i = 0; i < n_pts; ++i)
    {
      // Warp 3D point to left and right
      const Pixel &pt_l = pts_l1[i];
      const Pixel &pt_r = pts_r1[i];
      Point Xl = R10 * X[i] + t10;
      Point Xr = T_rl.block<3, 3>(0, 0) * Xl + T_rl.block<3, 1>(0, 3);

      // left
      float iz_l = 1.0f / Xl(2);
      float xiz_l = Xl(0) * iz_l;
      float yiz_l = Xl(1) * iz_l;
      float fxxiz_l = fx_l * xiz_l;
      float fyyiz_l = fy_l * yiz_l;

      Pixel pt_l_warp(fxxiz_l + cx_l, fyyiz_l + cy_l);
      float rx_l = pt_l_warp.x - pt_l.x;
      float ry_l = pt_l_warp.y - pt_l.y;

      // right
      float iz_r = 1.0f / Xr(2);
      float xiz_r = Xr(0) * iz_r;
      float yiz_r = Xr(1) * iz_r;
      float fxxiz_r = fx_r * xiz_r;
      float fyyiz_r = fy_r * yiz_r;

      Pixel pt_r_warp(fxxiz_r + cx_r, fyyiz_r + cy_r);
      float rx_r = pt_r_warp.x - pt_r.x;
      float ry_r = pt_r_warp.y - pt_r.y;

      // Huber weight calculation by the Manhattan distance
      float weight = 1.0f;
      bool flag_weight = false;
      float absrxry = abs(rx_l) + abs(ry_l) + abs(rx_r) + abs(ry_r);
      absrxry *= 0.5f;
      if (absrxry >= THRES_HUBER)
      {
        weight = THRES_HUBER / absrxry;
        flag_weight = true;
      }

      if (absrxry >= THRES_REPROJ_ERROR)
      {
        mask_inlier[i] = false;
        ++cnt_invalid;
      }
      else
        mask_inlier[i] = true;

      // JtWJ, JtWr for x
      Vec6 Jt;
      Jt.setZero();
      Mat66 JtJ_tmp;
      JtJ_tmp.setZero();

      // Left x
      Jt(0) = fx_l * iz_l;
      Jt(1) = 0.0f;
      Jt(2) = -fxxiz_l * iz_l;
      Jt(3) = -fxxiz_l * yiz_l;
      Jt(4) = fx_l * (1.0f + xiz_l * xiz_l);
      Jt(5) = -fx_l * yiz_l;

      float w_rx = weight * rx_l;
      float err = w_rx * rx_l;

      // JtWJ.noalias() += weight *(Jt*Jt.transpose());
      this->calcJtWJ_x(weight, Jt, JtJ_tmp);
      JtWJ.noalias() += JtJ_tmp;
      mJtWr.noalias() -= (w_rx)*Jt;
      err_curr += rx_l * rx_l;

      // Left y
      Jt(0) = 0.0f;
      Jt(1) = fy_l * iz_l;
      Jt(2) = -fyyiz_l * iz_l;
      Jt(3) = -fy_l * (1.0f + yiz_l * yiz_l);
      Jt(4) = fyyiz_l * xiz_l;
      Jt(5) = fy_l * xiz_l;

      float w_ry = weight * ry_l;
      err = w_ry * ry_l;

      this->calcJtWJ_y(weight, Jt, JtJ_tmp);
      JtWJ.noalias() += JtJ_tmp;
      mJtWr.noalias() -= (w_ry)*Jt;
      err_curr += ry_l * ry_l;

      // Right x
      Jt(0) = fx_r * iz_r;
      Jt(1) = 0.0f;
      Jt(2) = -fxxiz_r * iz_r;
      Jt(3) = -fxxiz_r * yiz_r;
      Jt(4) = fx_r * (1.0f + xiz_r * xiz_r);
      Jt(5) = -fx_r * yiz_r;

      w_rx = weight * rx_r;
      err = w_rx * rx_r;

      // JtWJ.noalias() += weight *(Jt*Jt.transpose());
      this->calcJtWJ_x(weight, Jt, JtJ_tmp);
      JtWJ.noalias() += JtJ_tmp;
      mJtWr.noalias() -= (w_rx)*Jt;
      err_curr += rx_r * rx_r;

      // Right y
      Jt(0) = 0.0f;
      Jt(1) = fy_r * iz_r;
      Jt(2) = -fyyiz_r * iz_r;
      Jt(3) = -fy_r * (1.0f + yiz_r * yiz_r);
      Jt(4) = fyyiz_r * xiz_r;
      Jt(5) = fy_r * xiz_r;

      w_ry = weight * ry_r;
      err = w_ry * ry_r;

      this->calcJtWJ_y(weight, Jt, JtJ_tmp);
      JtWJ.noalias() += JtJ_tmp;
      mJtWr.noalias() -= (w_ry)*Jt;
      err_curr += ry_r * ry_r;
    } // END FOR

    err_curr *= (inv_npts * 0.5f);
    err_curr = std::sqrt(err_curr);
    float delta_err = abs(err_curr - err_prev);

    // Solve H^-1*Jtr;
    for (size_t i = 0; i < 6; ++i)
      JtWJ(i, i) *= (1.0f + lambda); // lambda

    PoseSE3Tangent delta_xi = JtWJ.ldlt().solve(mJtWr);

    // Update matrix
    PoseSE3 dT;
    geometry::se3Exp_f(delta_xi, dT);
    T10_optimized.noalias() = dT * T10_optimized;

    err_prev = err_curr;
    // std::cout << "reproj. err. (avg): " << err_curr << ", step: " << delta_xi.transpose() << std::endl;
    if (delta_xi.norm() < THRES_DELTA_XI || delta_err < THRES_DELTA_ERROR)
    {
      std::cout << "poseonly Stereo BA stops at: " << iter << ", err: " << err_curr << ", derr: " << delta_err << ", deltaxi: " << delta_xi.norm() << ", # invalid: " << cnt_invalid << "\n";
      break;
    }
    if (iter == MAX_ITER - 1)
      std::cout << "!! WARNING !! poseonly Stereo BA stops at full iterations!!"
                << ", err: " << err_curr << ", derr: " << delta_err << ", # invalid: " << cnt_invalid << "\n";
  } // END iter

  if (!std::isnan(T10_optimized.norm()))
  {
    PoseSE3 T01_update;
    T01_update << geometry::inverseSE3_f(T10_optimized);
    T01 = T01_update;
  }
  else
  {
    std::cout << "!! WARNING !! poseonly BA yields NAN value!!"
              << ", T10_optimized: \n"
              << T10_optimized << "\n";
    is_success = false; // if nan, do not update.
  }

  return is_success;
};