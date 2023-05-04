#ifndef _DEPTH_FILTER_H_
#define _DEPTH_FILTER_H_
#include <iostream>
#include <cmath>

class DepthFilter
{
public:
  DepthFilter(){};
  ~DepthFilter(){};

public:
  void updateNormalDistribution(
      double x_prev, double cov_prev,
      double x_curr, double cov_curr,
      double &x_updated, double &cov_updated);

  void updateStudentTDistribution(
      double x_prev, double cov_prev, double a_prev, double b_prev, double x_min_prev, double x_max_prev,
      double x_curr, double cov_curr, double a_curr, double b_curr,
      double &x_updated, double &cov_updated, double &x_min_updated, double &x_max_updated);
};
#endif