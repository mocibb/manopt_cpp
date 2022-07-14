#ifndef PRODUCT_EUCLIDEAN_H_
#define PRODUCT_EUCLIDEAN_H_

#include "manifold.hpp"
#include <Eigen/Dense>
#include <utility>
#include <iostream>

namespace manopt {

template <class T1, class T2>
class ProductManifold;

namespace internal {
template <class T1, class T2>
struct traits<ProductManifold<T1, T2>> {
  using Scalar   = typename T1::Scalar;
  using MPoint   = std::pair<typename T1::MPoint,   typename T2::MPoint>; 
  using TVector  = std::pair<typename T1::TVector,  typename T2::TVector>; 
  using EPoint   = std::pair<typename T1::EPoint,   typename T2::EPoint>; 
  using ETVector = std::pair<typename T1::ETVector, typename T2::ETVector>;
};
}

template <class T1, class T2>
class ProductManifold : public AbstractManifold<ProductManifold<T1, T2>> {
    using Base = AbstractManifold<ProductManifold<T1, T2>>;
  public:
    using Scalar   = typename Base::Scalar;
    using MPoint   = typename Base::MPoint;
    using TVector  = typename Base::TVector;
    using ETVector = typename Base::ETVector;
    using SubType1 = T1;
    using SubType2 = T2;

    typedef std::shared_ptr<ProductManifold<T1, T2>> Ptr;  

    ProductManifold(T1 t1_, T2 t2_) : 
      t1(t1_), t2(t2_)
    {}

    int dim() const override { 
      return t1.dim()+t2.dim(); 
    }

    MPoint rand() const override {     
      return std::make_pair(t1.rand(), t2.rand());
    }

    TVector randvec(const MPoint& x) const override {      
      return std::make_pair(t1.randvec(x.first), t2.randvec(x.second));
    }

    TVector zerovec(const MPoint& x) const override {
      return std::make_pair(t1.zerovec(x.first), t2.zerovec(x.second));
    }        

    Scalar inner(const MPoint& x, const TVector& d1, const TVector& d2) const override {
      return t1.inner(x.first, d1.first, d2.first) + t2.inner(x.second, d1.second, d2.second);
    }

    Scalar norm(const MPoint& x, const TVector& d) const override {
      return sqrt(inner(x, d, d));
    }

    TVector proj(const MPoint& x, const ETVector& H) const override {
      return std::make_pair(t1.proj(x.first, H.first), t2.proj(x.second, H.second));
    }

    TVector tangent(const MPoint& x, const ETVector& H) const override {
      return std::make_pair(t1.tangent(x.first, H.first), t2.tangent(x.second, H.second));
    }    

    MPoint retr(const MPoint& x, const TVector& U, Scalar t = 1) const override {
      return std::make_pair(t1.retr(x.first, U.first, t), t2.retr(x.second, U.second, t));
    }

    TVector transp(const MPoint& x1, const MPoint& x, const TVector& U) const override {
      return std::make_pair(t1.transp(x1.first, x.first, U.first), t2.transp(x1.second, x.second, U.second));
    }

    TVector lincomb(const MPoint&x, Scalar a, const TVector& d) const override {
      return std::make_pair(t1.lincomb(x.first, a, d.first), 
                            t2.lincomb(x.second, a, d.second));
    }    

    TVector lincomb(const MPoint&x, Scalar a1, const TVector& d1, Scalar a2, const TVector& d2) const override {
      return std::make_pair(t1.lincomb(x.first, a1, d1.first, a2, d2.first), 
                            t2.lincomb(x.second, a1, d1.second, a2, d2.second));
    }

    Scalar typicaldist() const override {
      return sqrt(std::pow(t1.typicaldist(), 2) + std::pow(t2.typicaldist(), 2));
    }  
    
  //private: => public
    T1 t1;
    T2 t2;
};

}  // namespace manopt


#endif  // PRODUCT_EUCLIDEAN_H_
