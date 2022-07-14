#ifndef MANIFOLD_H_
#define MANIFOLD_H_

#include <memory>

namespace manopt {

const int Dynamic = -1;

namespace internal {
  template<class Derived>
  struct traits;
}

template <typename Derived>
class AbstractManifold {
 public:
  using Scalar = typename manopt::internal::traits<Derived>::Scalar;
  using MPoint = typename manopt::internal::traits<Derived>::MPoint;
  using TVector = typename manopt::internal::traits<Derived>::TVector;
  using EPoint = typename manopt::internal::traits<Derived>::EPoint;
  using ETVector = typename manopt::internal::traits<Derived>::ETVector;

  typedef std::shared_ptr<AbstractManifold<Derived>> Ptr;  

  virtual int dim() const = 0;

  //流形上不好定义zero，所以rand也用来初始化。
  virtual MPoint rand() const = 0;

  virtual TVector randvec(const MPoint& x) const = 0;

  virtual TVector zerovec(const MPoint& x) const = 0;
  
  virtual Scalar inner(const MPoint& x, const TVector& d1, const TVector& d2) const = 0;

  virtual Scalar norm(const MPoint& x, const TVector& d) const = 0;

  virtual TVector proj(const MPoint& x, const ETVector& H) const = 0;

  //H是embed上的表示
  virtual TVector tangent(const MPoint& x, const ETVector& H) const = 0;

  virtual MPoint retr(const MPoint& x, const TVector& U, Scalar t = 1) const = 0;

  //把x1点处的切向量U移动到x点。
  virtual MPoint transp(const MPoint& x1, const MPoint& x, const TVector& U) const = 0;

  virtual TVector lincomb(const MPoint&x, Scalar a, const TVector& d) const = 0;

  virtual TVector lincomb(const MPoint&x, Scalar a1, const TVector& d1, Scalar a2, const TVector& d2) const = 0;

  // returns the typical distance on the Manifold M, 
  // which is for example the longest distance in a unit cell or injectivity radius.
  virtual Scalar typicaldist() const = 0;
};

template <typename Derived>
class MatrixManifold : public AbstractManifold<Derived> {
  public:
    enum {
      Rows = internal::traits<Derived>::Rows,
      Cols = internal::traits<Derived>::Cols,
    };
};

}  // namespace manopt

#endif  // MANIFOLD_H_
