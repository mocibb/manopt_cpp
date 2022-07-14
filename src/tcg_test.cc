#include "rotation.hpp"
#include "problem.hpp"
#include "cost_func.hpp"
#include "tcg.hpp"
#include <iostream>
#include <iomanip>
#include <Eigen/Core>

using namespace manopt;

using MType = Rotation<double, 3, 1>;

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

    Scalar Cost(const MPoint& Rx) {
      Scalar err = 0;
      for (int i=0; i<xx.size(); i++) {
        err += (yy[i]-Rx*xx[i]).squaredNorm();
      }
      
      return err;
    }

    TVector Gradient(const MPoint& Rx) {
      TVector grad = TVector::Zero();
      TVector K = TVector::Zero();
      for (int i=0; i<xx.size(); i++) {
        K += Rx.transpose()*yy[i]*xx[i].transpose();
      }

      for (int i=0; i<3; i++) {
        Eigen::Vector3d bv = Eigen::Vector3d::Zero();
        bv[i] = 1;
        TVector base = sqrt(2)/2*manifold->hat(bv);
        grad += 2*manifold->inner(Rx, base, K)*base;
      }
      return -grad;
    }
      
  private:
    MPtr manifold;
    std::vector<Eigen::Vector3d> xx;
    std::vector<Eigen::Vector3d> yy;
    
};

int main() {
  MType::Ptr M = std::make_shared<MType>();
  typedef MType::MPoint MPoint;
  typedef MType::TVector TVector;

  Problem<MType>::Ptr problem = std::make_shared<Problem<MType>>();
  problem->SetManifold(M);

  std::vector<Eigen::Vector3d> xx;
  std::vector<Eigen::Vector3d> yy;

  auto x = M->rand();
  int N=10;
  //std::cout << std::fixed << std::setprecision(10) << "rot = " << x << std::endl;
  for (int i=0; i<N; i++) {
    xx.push_back(10*Eigen::Vector3d::Random());
    //std::cout << std::fixed << std::setprecision(10) << xx.back().transpose() << std::endl;
    yy.push_back(x * xx.back());
    //std::cout << yy.back().transpose() << std::endl;
  }

  std::shared_ptr<GradientCostFunction<MType>> func = std::make_shared<ICPCostFunction>(M, xx , yy);
  problem->SetGradientCostFunction(func);

  // // auto* minimizer = Minimizer::Create(MinimizerType::TRUST_REGION);

  // // MType x=M.rand();
  // // minimizer.Minimize(M, problem, x);

  TruncatedConjugateGradient<MType> tCG(problem, 1, 0.1, false, 1, 3);
  MPoint xp;
  // xp << -0.556901566919582,   0.569921486505996,   0.604193796675560,
  //        0.119423032618585,   0.774822686777889,  -0.620796217149530,
  //       -0.821948163876581,  -0.273567730618556,  -0.499561720666789;

  xp << -0.610781863867384,   0.754507303162354,   0.240133804878356,
         0.215023383875814,   0.449933313925957,  -0.866790030749201,
        -0.762043607123162,  -0.477785247254745,  -0.437047821580708;

  TVector grad;
  // grad << 0, 37.2910299761626, -168.4016299926723,
  //        -37.2910299761626, 0, 636.6724794533905,
  //         168.4016299926723, -636.6724794533905, 0;
  grad <<    0,   93.1050313081506,  -63.9490650111049,
            -93.1050313081506, 0, 407.9876545622247,
             63.9490650111049, -407.9876545622247, 0;

  TVector eta = TVector::Zero();

  tCG.Solve(xp, grad, eta, 1.360349523175663);
  // , TVector grad, TVector eta, double tr_radius

  return 0;
}