#include "rotation.hpp"
#include "problem.hpp"
#include "cost_func.hpp"
#include "tcg.hpp"
#include "trust_region.hpp"
#include <iostream>
#include <iomanip>
#include <Eigen/Core>
#include <ctime>

using namespace manopt;

using MType = RotationX<double, 3>;

class ICPCostFunction : public GradientCostFunction<MType> {
  public:
    using Scalar  = typename MType::Scalar;
    using MPoint  = typename MType::MPoint;
    using TVector = typename MType::TVector;
    using MPtr    = typename MType::Ptr;

    ICPCostFunction(const MPtr& manifold_,
                    const std::vector<Eigen::Vector3d>& xx_, 
                    const std::vector<Eigen::Vector3d>& yy_) :
      manifold(manifold_), xx(xx_), yy(yy_) {}

    Scalar cost(const MPoint& Rx) const override {
      Scalar err = 0;
      for (uint i=0; i<xx.size(); i++) {
        err += (yy[i]-Rx*xx[i]).squaredNorm();
      }
      
      return err;
    }

    TVector gradient(const MPoint& Rx) const override {
      TVector grad = manifold->zerovec(Rx);
      TVector K = manifold->zerovec(Rx);
      for (uint i=0; i<xx.size(); i++) {
        K += Rx.transpose()*yy[i]*xx[i].transpose();
      }

      for (int i=0; i<3; i++) {
        Eigen::Vector3d bv = Eigen::Vector3d::Zero();
        bv[i] = 1;
        TVector base = sqrt(2)/2*manifold->hat(bv);
        //grad += 2*manifold->inner(Rx, base, K)*base;
        grad += 2*(base*K).trace()*base;
      }
      return grad;
    }
      
  private:
    MPtr manifold;
    std::vector<Eigen::Vector3d> xx;
    std::vector<Eigen::Vector3d> yy;
    
};

int main() {
  MType::Ptr M = std::make_shared<MType>(1);
  typedef MType::MPoint MPoint;

  Problem<MType>::Ptr problem = std::make_shared<Problem<MType>>();
  problem->setManifold(M);

  std::vector<Eigen::Vector3d> xx;
  std::vector<Eigen::Vector3d> yy;

  auto x = M->rand();
  int N=10;
  srand (2);
  //std::cout << std::fixed << std::setprecision(10) << "rot = " << x << std::endl;
  for (int i=0; i<N; i++) {
    xx.push_back(10*Eigen::Vector3d::Random());
    //std::cout << std::fixed << std::setprecision(10) << xx.back().transpose() << std::endl;
    yy.push_back(x * xx.back());
    //std::cout << yy.back().transpose() << std::endl;
  }

  std::shared_ptr<GradientCostFunction<MType>> func = std::make_shared<ICPCostFunction>(M, xx , yy);
  problem->setGradientCostFunction(func);

  MPoint xp = M->rand();

  xp << -0.556901566919582,   0.569921486505996,   0.604193796675560,
         0.119423032618585,   0.774822686777889,  -0.620796217149530,
        -0.821948163876581,  -0.273567730618556,  -0.499561720666789;

  TrustRegion<MType> tr(problem);
  Summary summary;
  tr.solve(xp, &summary);
  std::cout << xp << std::endl;
  std::cout << summary.fullReport() << std::endl;

  return 0;
}