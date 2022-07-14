#ifndef SPHERE_H_
#define SPHERE_H_

#include "manifold.hpp"
#include <cmath>
#include <Eigen/Dense>

namespace manopt {

template <class Scalar_, int N>
class Sphere;

namespace internal {
template <class Scalar_, int N>
struct traits<Sphere<Scalar_, N>> {
  using Scalar   = Scalar_;
  using MPoint   = Eigen::Matrix<Scalar_, N, 1>; 
  using TVector  = MPoint;
  using EPoint   = MPoint;
  using ETVector = MPoint;
};
}
  
template <class Scalar_, int N>
class Sphere : public AbstractManifold<Sphere<Scalar_, N>> {
    using Base = AbstractManifold<Sphere<Scalar_, N>>;
  public:
    using Scalar   = typename Base::Scalar;
    using MPoint   = typename Base::MPoint;
    using TVector  = typename Base::TVector;
    using ETVector = typename Base::ETVector;
    using VectorX  = typename Eigen::Matrix<Scalar, 1, Eigen::Dynamic>;

    typedef std::shared_ptr<Sphere<Scalar_, N>> Ptr;  

    int dim() const override { 
      return N-1; 
    }

    MPoint rand() const override {
      MPoint x = MPoint::Random();
      return x / x.norm();
    }

    TVector randvec(const MPoint& x) const {
      TVector d = proj(x, TVector::Random());
      return d / d.norm();
    }

    TVector zerovec(const MPoint& x) const override {
      return TVector::Zero();
    }    

    Scalar inner(const MPoint& x, const TVector& d1, const TVector& d2) const override {
      Eigen::Map<const VectorX> v1(d1.data(), d1.size());
      Eigen::Map<const VectorX> v2(d2.data(), d2.size());
      return v1.dot(v2);
    }

    Scalar norm(const MPoint& x, const TVector& d) const override {
      Eigen::Map<const VectorX> v(d.data(), d.size());
      return v.norm();
    }

    TVector proj(const MPoint& x, const ETVector& H) const override {
      Eigen::Map<const VectorX> x1(x.data(), x.size());
      Eigen::Map<const VectorX> d1(H.data(), H.size());

      return H - x1.dot(d1)*x;
    }

    TVector tangent(const MPoint& x, const ETVector& H) const override {
      return proj(x, H);
    }    

    MPoint retr(const MPoint& x, const TVector& U, Scalar t = 1) const override {    
      MPoint y = x + t*U;
      return y / y.norm();
    }

    TVector transp(const MPoint& x1, const MPoint& x, const TVector& U) const override {
      return proj(x, U);
    }

    TVector lincomb(const MPoint&x, Scalar a, const TVector& d) const override {
      return a*d;
    }

    TVector lincomb(const MPoint&x, Scalar a1, const TVector& d1, Scalar a2, const TVector& d2) const override {
      return a1*d1+a2*d2;
    }

    Scalar typicaldist() const override {
      return M_PI;
    } 

};

}  // namespace manopt

#endif  // SPHERE_H_
