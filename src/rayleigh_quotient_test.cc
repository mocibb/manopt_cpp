#include "cost_func.hpp"
#include "problem.hpp"
#include "sphere.hpp"
#include "trust_region.hpp"
#include "conjugate_gradient.hpp"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <ctime>
#include <iomanip>
#include <iostream>

using namespace manopt;
int constexpr N = 1000;
using MType = Sphere<double, N>;

class RQCostFunction : public GradientCostFunction<MType> {
 public:
  using Scalar = typename MType::Scalar;
  using MPoint = typename MType::MPoint;
  using TVector = typename MType::TVector;
  using MPtr = typename MType::Ptr;

  RQCostFunction(const MPtr& manifold_, const Eigen::MatrixXd& A_)
      : manifold(manifold_), A(A_) {}

  Scalar cost(const MPoint& x) const override {
    Eigen::MatrixXd v = -x.transpose() * A * x;
    return v(0, 0);
  }

  TVector gradient(const MPoint& x) const override {
    TVector grad = -2 * A * x;
    return manifold->proj(x, grad);
  }

 private:
  MPtr manifold;
  Eigen::MatrixXd A;
};

int main() {
  MType::Ptr M = std::make_shared<MType>();
  typedef MType::MPoint MPoint;

  for (int i = 0; i < 10; i++) {
    srand(i);
    std::cout << " i  = " << i << std::endl;
    Eigen::MatrixXd B = Eigen::MatrixXd::Random(N, N);
    Eigen::MatrixXd A = 0.5 * (B.transpose() + B);

    // A << 0.5377, 1.3480, -1.3462, 1.3480, 0.3188, -0.4825, -1.3462, -0.4825,
    //     3.5784;

    // std::cout << "A = " << A << std::endl;
    Problem<MType>::Ptr problem = std::make_shared<Problem<MType>>();
    problem->setManifold(M);

    std::shared_ptr<GradientCostFunction<MType>> func =
        std::make_shared<RQCostFunction>(M, A);
    problem->setGradientCostFunction(func);

    MPoint x0 = M->rand();
    // std::cout << x0 << std::endl;

    // std::cout << "x_old = " << x0 << std::endl;
    ConjugateGradient<MType> cg(problem);
    TrustRegion<MType> tr(problem);

    Summary summary;
    double start = wallTimeInSeconds();
    MPoint x1 = x0;
    tr.solve(x1, &summary);
    std::cout << wallTimeInSeconds() - start << std::endl;
    start = wallTimeInSeconds();
    MPoint x2 = x0;
    cg.solve(x2, &summary);
    std::cout << wallTimeInSeconds() - start << std::endl;
    if ((x2-x1).norm() > 1e-3) {
      throw std::runtime_error("> 1e-3");
    }

    // if (!summary.isSolutionUsable()) {
    //   break;
    // }
  }


  return 0;
}