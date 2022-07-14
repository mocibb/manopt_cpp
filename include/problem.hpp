#ifndef PROBLEM_H_
#define PROBLEM_H_

#include <cmath>
#include <iostream>
#include <iomanip>
#include <memory>
#include <limits>
#include <unordered_map>
#include "manifold.hpp"
#include "cost_func.hpp"
#include <glog/logging.h>

namespace manopt
{
  template <typename MType>
  class Problem
  {
  public:
    using Scalar = typename MType::Scalar;
    using MPoint = typename MType::MPoint;
    using TVector = typename MType::TVector;
    using MPtr = typename MType::Ptr;

    typedef std::shared_ptr<Problem<MType>> Ptr;

  private:
    MPtr manifold;
    typename GradientCostFunction<MType>::Ptr functor_;
    GradientCostFunction<MType> *func_ptr_;

  public:
    Problem() : func_ptr_(nullptr){};
    Problem(Problem &&);
    Problem &operator=(Problem &&){};

    Problem(const Problem &) = delete;
    Problem &operator=(const Problem &) = delete;

    ~Problem(){};

    void setManifold(MPtr manifold_)
    {
      manifold = manifold_;
    }

    MPtr M() const
    {
      return manifold;
    }

    //
    TVector getGradient(const MPoint &x)
    {
      CHECK(functor_.get() != nullptr || func_ptr_ != nullptr) << "must set setGradientCostFunction[Ptr] first!";
      return func_ptr_ != nullptr ? func_ptr_->gradient(x) : functor_->gradient(x);
    }

    //实现数值Hessian
    TVector getHessian(const MPoint &x, const TVector &d)
    {
      Scalar norm_d = M()->norm(x, d);
      Scalar eps = (Scalar)std::pow(2, -14);
      if (norm_d < std::numeric_limits<Scalar>::epsilon())
      {
        std::cout << "!!!norm_d is too small, return zerovec!!!" << std::endl;
        return M()->zerovec(x);
      }

      Scalar c = eps / norm_d;
      TVector grad = getGradient(x);

      MPoint x1 = M()->retr(x, d, c);
      TVector grad1 = getGradient(x1);

      grad1 = M()->transp(x1, x, grad1);
      return M()->lincomb(x, 1 / c, grad1, -1 / c, grad);
    }

    void setGradientCostFunction(typename GradientCostFunction<MType>::Ptr func)
    {
      functor_ = func;
    }

    void setGradientCostFunctionPtr(GradientCostFunction<MType> *func_ptr)
    {
      func_ptr_ = func_ptr;
    }

    Scalar getCost(const MPoint &x)
    {
      CHECK(functor_.get() != nullptr || func_ptr_ != nullptr) << "must set setGradientCostFunction[Ptr] first!";
      return func_ptr_ != nullptr ? func_ptr_->cost(x) : functor_->cost(x);
    }

    std::pair<Scalar, TVector> getCostAndGrad(const MPoint &x)
    {
      return std::make_pair(getCost(x), getGradient(x));
    }

    void iterationCallback(const MPoint &x)
    {
      CHECK(functor_.get() != nullptr || func_ptr_ != nullptr) << "must set setGradientCostFunction[Ptr] first!";
      return func_ptr_ != nullptr ? func_ptr_->iterationCallback(x) : functor_->iterationCallback(x);
    }
  };

} // namespace manopt

#endif // PROBLEM_H_
