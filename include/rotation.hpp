#ifndef ROTATION_H_
#define ROTATION_H_

#include "manifold.hpp"
#include <cmath>
#include <iostream>
#include <Eigen/Dense>
#include <glog/logging.h>

namespace manopt {

template <class Scalar_, int N, int k>
class Rotation;

namespace internal {
template <class Scalar_, int N, int k>
struct traits<Rotation<Scalar_, N, k>> {
  enum {
    Rows = N,
    Cols = (k==Dynamic ? Dynamic : N*k)
  };

  using Scalar   = Scalar_;
  using MPoint   = Eigen::Matrix<Scalar_, Rows, Cols>; 
  using TVector  = MPoint;
  using EPoint   = MPoint;
  using ETVector = MPoint;
};
}

template <class Scalar_, int N, int k>
class Rotation : public MatrixManifold<Rotation<Scalar_, N, k>> {
    using Base = MatrixManifold<Rotation<Scalar_, N, k>>;
  public:
    using Scalar   = typename Base::Scalar;
    using MPoint   = typename Base::MPoint;
    using MSPoint  = Eigen::Matrix<Scalar_, N, N>;
    using TVector  = typename Base::TVector;
    using ETVector = typename Base::ETVector;
    using VectorX  = typename Eigen::Matrix<Scalar, 1, Eigen::Dynamic>;

    typedef std::shared_ptr<Rotation<Scalar_, N, k>> Ptr;  

    Rotation(int sz = k) : sz_(sz) {
      CHECK(sz != Dynamic);
    }

    int dim() const override { 
      return sz_*N*(N-1)/2; 
    }
    
    //Generated as such, Q is uniformly distributed over O(n),
    //the group of orthogonal matrices; see Mezzadri 2007
    MPoint rand() const override {
      MPoint Q = MPoint::Random(N, cols());
      for (int i=0; i<sz_; i++) {
        MSPoint rand = MSPoint::Random();
        Eigen::HouseholderQR<MSPoint> qr(rand);
        MSPoint QS = qr.householderQ();
        if (QS.determinant() < 0) {
          const VectorX col0 = QS.col(0);
          QS.col(0) = QS.col(1);
          QS.col(1) = col0;
        } 
        Q.block(0, i*N, N, N) = QS;
      }
     
      return Q;
    }

    TVector randvec(const MPoint& x) const {
      TVector U(N, cols());
      for (int i=0; i<sz_; i++) {
        MSPoint US = MSPoint::Random();
        U.block(0, i*N, N, N) = US-US.transpose();
      }
      return U / U.norm();
    }

    TVector zerovec(const MPoint& x) const override {
      return TVector::Zero(N, cols());
    }    

    MSPoint at(const MPoint& x, int idx) const {
      CHECK(idx < sz_);
      return x.block(0, idx*N, N, N);
    }

    void set(MPoint& x, int idx, const MSPoint& v) const {
      CHECK(idx < sz_);
      x.block(0, idx*N, N, N) = v;
    }    

    void accumulate(MPoint& x, int idx, const MSPoint& v) const {
      CHECK(idx < sz_);
      x.block(0, idx*N, N, N) += v;
    }

    Scalar inner(const MPoint& x, const TVector& d1, const TVector& d2) const override {
      Eigen::Map<const VectorX> v1(d1.data(), d1.size());
      Eigen::Map<const VectorX> v2(d2.data(), d2.size());
      return v1.dot(v2);
    }

    Scalar norm(const MPoint& x, const TVector& d) const override {
      return d.norm();
    }

    TVector proj(const MPoint& x, const ETVector& H) const override {
      TVector U(N, cols());
      for (int i=0; i<sz_; i++) {
        MSPoint US = x.block(0, i*N, N, N).transpose()*H.block(0, i*N, N, N);
        U.block(0, i*N, N, N) = 0.5*(US-US.transpose());
      }
      return U;
    }

    TVector tangent(const MPoint& x, const ETVector& H) const override {
      TVector U(N, cols());
      for (int i=0; i<sz_; i++) {
        MSPoint HR = H.block(0, i*N, N, N);
        U.block(0, i*N, N, N) = 0.5*(HR-HR.transpose());
      }
      return U;
    }    

    MPoint retr(const MPoint& x, const TVector& U, Scalar t = 1) const override {
      MPoint Q(N, cols());
      for (int i=0; i<sz_; i++) {
        MSPoint xu = x.block(0, i*N, N, N)*U.block(0, i*N, N, N);
        Eigen::HouseholderQR<MSPoint> qr(x.block(0, i*N, N, N)+t*xu);
        MSPoint QS = qr.householderQ();
        MSPoint R =  qr.matrixQR();
        for (int j=0; j<N; j++) {
          QS.col(j) = (R(j, j) < 0 ? -1 : 1)*QS.col(j);
        }
        Q.block(0, i*N, N, N) = QS;
      }
      return Q;
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
      return M_PI*sqrt(N*sz_);
    }    

    TVector hat(const VectorX& v) const {
      TVector U = TVector::Zero(N, cols());

      if (N == 2) {
        U(0, 1) = v[0];
        U(1, 0) = -v[0];
      } else {
        U(0, 1) = -v[2];
        U(0, 2) = v[1];
        U(1, 0) = v[2];
        U(1, 2) = -v[0];
        U(2, 0) = -v[1];
        U(2, 1) = v[0];

        if (N>3) {
          throw new std::runtime_error("not supported!");
        }

        // if (N > 3) {
        //   int n = 3;
        //   for (int i=3; i<N-1; i++) {
        //     for (int j=0; j<i; j++) {
        //       //符号可能有问题
        //       U[i, j] = v[n];
        //       U[j, i] = -v[n];
        //       n += 1;
        //     }
        //   }
        // }
      }
      return U;
    }

  private:
    inline int cols() const { return N*sz_; }
    int sz_;
};

template <class Scalar_, int N>
using RotationX = Rotation<Scalar_, N, Dynamic>;

}  // namespace manopt

#endif  // ROTATION_H_
