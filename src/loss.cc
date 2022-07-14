#include "loss.hpp"
#include <cmath>
#include <limits>
#include <algorithm>

namespace manopt {

double TrivialLOSS::v(double v) const {
  return v;
}

double TrivialLOSS::j(double v) const {
  return 1.0;
}

double HuberLOSS::v(double v) const {
  if (v > b_) {
    return 2.0 * a_ * sqrt(v) - b_;
  } else {
    return v;
  }
}

double HuberLOSS::j(double v) const {
  if (v > b_) {
    return std::max(std::numeric_limits<double>::min(), a_ / sqrt(v));
  } else {
    return 1.0;
  }
}

double CauchyLOSS::v(double v) const {
  return b_*log(1.0 + v * c_);
}

double CauchyLOSS::j(double v) const {
  const double inv = 1.0 / (1.0 + v * c_);    
  return std::max(std::numeric_limits<double>::min(), inv);
}

}  // namespace manopt