# manopt_cpp（c++ solver for optimization on manifolds）

流形上的求解器

manopt_cpp是[manopt](https://www.manopt.org/) 的C++版本，用来求解流形上的优化问题。

manopt_cpp使用了模板静态存储流形数据结构，避免了频繁申请内存带来的效率降低。


支持的流形

1）欧式空间
2）球面
3）旋转矩阵
4）积流形

支持的求解器

目前支持两种求解器，Trust-regions (RTR)和Conjugate-gradient。


求解最大特征值问题 $\max\limits_{x\in\mathbb{R}^n, x \neq 0} \frac{x^\top A x}{x^\top x}.$

问题定义

```c++
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
```

求解
```c++
MType::Ptr M = std::make_shared<MType>();
typedef MType::MPoint MPoint;

Eigen::MatrixXd B = Eigen::MatrixXd::Random(N, N);
Eigen::MatrixXd A = 0.5*(B.transpose() + B);

Problem<MType>::Ptr problem = std::make_shared<Problem<MType>>();
problem->setManifold(M);

std::shared_ptr<GradientCostFunction<MType>> func = std::make_shared<RQCostFunction>(M, A);
problem->setGradientCostFunction(func);

MPoint x0 = M->rand();

TrustRegion<MType> tr(problem);
Summary summary;
double start = wallTimeInSeconds();
tr.solve(x0, &summary);
std::cout << wallTimeInSeconds() - start << std::endl;
std::cout << summary.fullReport() << std::endl;
```
