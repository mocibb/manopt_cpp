#ifndef EUCLIDEAN_H_
#define EUCLIDEAN_H_

#include "manifold.hpp"
#include <Eigen/Dense>

namespace manopt {

template <class Scalar_, int N, int k>
class Euclidean;

namespace internal {
template <class Scalar_, int N, int k>
struct traits<Euclidean<Scalar_, N, k>> {
  enum {
    Rows = N,
    Cols = (k==Dynamic ? Dynamic : k)
  };

  using Scalar   = Scalar_;
  using MPoint   = Eigen::Matrix<Scalar_, Rows, Cols>; 
  using TVector  = MPoint;
  using EPoint   = MPoint;
  using ETVector = MPoint;
};
}

template <class Scalar_, int N, int k>
class Euclidean : public MatrixManifold<Euclidean<Scalar_, N, k>> {
    using Base = MatrixManifold<Euclidean<Scalar_, N, k>>;
  public:
    using Scalar   = typename Base::Scalar;
    using MPoint   = typename Base::MPoint;
    using MSPoint  = Eigen::Matrix<Scalar_, N, 1>;
    using TVector  = typename Base::TVector;
    using ETVector = typename Base::ETVector;
    using VectorX  = typename Eigen::Matrix<Scalar, 1, Eigen::Dynamic>;

    Euclidean(int sz = k) : sz_(sz) {
      CHECK(sz != Dynamic);
    }

    int dim() const override { 
      return N*sz_; 
    }

    MPoint rand() const override {     
      return MPoint::Random(N, sz_);
    }

    TVector randvec(const MPoint& x) const override {
      TVector U = TVector::Random(N, sz_);      
      return U / U.norm();
    }

    TVector zerovec(const MPoint& x) const override {
      return TVector::Zero(N, sz_);
    }    

    MSPoint at(const MPoint& x, int idx) const {
      CHECK(idx < sz_);
      return x.col(idx);
    }    

    void set(MPoint& x, int idx, const MSPoint& v) const {
      CHECK(idx < sz_);
      x.col(idx) = v;
    }        

    void accumulate(MPoint& x, int idx, const MSPoint& v) const {
      CHECK(idx < sz_);
      x.col(idx) += v;
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
      return H;
    }

    TVector tangent(const MPoint& x, const ETVector& H) const override {
      return H;
    }

    MPoint retr(const MPoint& x, const TVector& U, Scalar t = 1) const override {
      return x+t*U;
    }

    TVector transp(const MPoint& x1, const MPoint& x, const TVector& U) const override {
      return U;
    }

    TVector lincomb(const MPoint&x, Scalar a, const TVector& d) const override {
      return a*d;
    }
    
    TVector lincomb(const MPoint&x, Scalar a1, const TVector& d1, Scalar a2, const TVector& d2) const override {
      return a1*d1+a2*d2;
    }

    Scalar typicaldist() const override {
      return sqrt(dim());
    }        

  private:
    int sz_;
};

template <class Scalar_, int N>
using EuclideanX = Euclidean<Scalar_, N, Dynamic>;

}  // namespace manopt

#endif  // EUCLIDEAN_H_
