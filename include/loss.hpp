#ifndef LOSS_H_
#define LOSS_H_

namespace manopt {

class LOSS {
 public:
  virtual ~LOSS() {}

  virtual double v(double v) const = 0;

  virtual double j(double v) const = 0;
};

class TrivialLOSS : public LOSS {
 public:
  explicit TrivialLOSS() {}

  double v(double v) const override;

  double j(double v) const override; 
};

class HuberLOSS : public LOSS {
 public:
  explicit HuberLOSS(double a) : a_(a), b_(a * a) {}

  double v(double v) const override;

  double j(double v) const override;  

 private:
  const double a_;
  // b = a^2.
  const double b_;
};

class CauchyLOSS : public LOSS {
 public:
  explicit CauchyLOSS(double a) : b_(a * a), c_(1 / b_) {}  

  double v(double v) const override;

  double j(double v) const override;  

 private:
  // b = a^2.
  const double b_;
  // c = 1 / a^2.
  const double c_;
};

}  // namespace manopt

#endif  // LOSS_H_
