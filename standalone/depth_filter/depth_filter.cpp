#include "depth_filter/depth_filter.h"

void DepthFilter::updateNormalDistribution(
    double x_prev, double cov_prev,
    double x_curr, double cov_curr,
    double &x_updated, double &cov_updated)
{
  // plus minus: 2, mult: 5, div: 1
  double inv_cov_sum = 1.0/(cov_prev + cov_curr);
  
  cov_updated = (cov_prev * cov_curr) * inv_cov_sum;
  x_updated = (x_prev * cov_curr + x_curr * cov_prev) * inv_cov_sum;
}

void DepthFilter::updateStudentTDistribution(
    double x_prev, double cov_prev, double a_prev, double b_prev, double x_min_prev, double x_max_prev,
    double x_curr, double cov_curr, double a_curr, double b_curr,
    double &x_updated, double &cov_updated, double& x_min_updated, double& x_max_updated)
{
  double inv_apb = 1.0/(a_prev + b_prev);
  double x_range = (x_max_prev - x_min_prev);

  double C1 = a_prev * inv_apb * 1.0 / std::sqrt(2.0 * 3.141592) / sigma * std::exp(-(x_curr - x_prev)*(x_curr - x_prev)/(2.0 * cov_prev));
  double C2 = b_prev * inv_apb / x_range;
  
  double invC1pC2 = 1.0 / (C1 + C2);
  C1 *= invC1pC2; // normalize
  C2 *= invC1pC2;

  double s = sqrt( 1.0/ (1.0/pow(seed->sigma,2) + 1.0/pow(tau_,2) ) );
  double m = s*s *(seed->mu / pow(seed->sigma,2) + x_ / pow(tau_,2) );

  double mu_new = C1*m + C2*seed->mu;
  double sigma_new = sqrt( C1*(pow(s,2) + pow(m,2)) + C2*(seed->sigma*seed->sigma + seed->mu*seed->mu) - mu_new*mu_new );

  double F = C1 * (seed->a + 1.0)/(seed->a + seed->b + 1.0) + C2*seed->a/(seed->a + seed->b + 1.0);
  double E = C1 * (seed->a + 1.0)/(seed->a + seed->b + 1.0)* (seed->a +2.0)/(seed->a + seed->b + 2.0) + C2 * seed->a / (seed->a + seed->b + 1.0)*(seed->a + 1.0)/(seed->a + seed->b + 2.0);

//  double a_new = (E-F)/(F-E/F);
//  double b_new = a_new * (1-F)/F;

  seed->a = (E-F)/(F-E/F);
  seed->b = seed->a*(1-F)/F;
  x_updated = mu_new;
  cov_updated = sigma_new;
}
