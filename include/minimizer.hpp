#ifndef MINIMIZER_H_
#define MINIMIZER_H_

#include <memory>
#include <string>
#include <vector>
#include "tcg.hpp"

namespace manopt
{

  enum TCGStopReason;

  enum TerminationType
  {
    // Minimizer terminated because one of the convergence criterion set
    // by the user was satisfied.
    CONVERGENCE,

    // The solver ran for maximum number of iterations or maximum amount
    // of time specified by the user, but none of the convergence
    // criterion specified by the user were met. The user's parameter
    // blocks will be updated with the solution found so far.
    NO_CONVERGENCE,

    // The minimizer terminated because of an error.  The user's
    // parameter blocks will not be updated.
    FAILURE,
  };

  struct IterationSummary
  {
    IterationSummary()
        : iteration(0),
          step_is_accept(false),
          cost(0.0),
          cost_change(0.0),
          gradient_norm(0.0),
          trust_region_ratio(0.0),
          trust_region_radius(0.0),
          consecutive_TRplus(0),
          consecutive_TRminus(0),
          used_cauchy(false),
          tcg_iterations(0),
          tcg_time(0.0),
          tcg_reason(),
          total_time(0.0) {}

    // 迭代次数
    int iteration;

    // step是否成功
    bool step_is_accept;

    // Cost
    double cost;

    // old_cost - new_cost 
    double cost_change;

    // 2-norm of the gradient vector.
    double gradient_norm;

    //用来判断二阶近似是否准确
    double trust_region_ratio;

    // 信赖域半径
    double trust_region_radius;

    //信赖半径增加次数
    int consecutive_TRplus;

    //信赖半径减小次数
    int consecutive_TRminus;

    //是否使用柯西点
    bool used_cauchy;

    // TCG迭代次数
    int tcg_iterations;

    // TCG时间
    double tcg_time;

    // TCG停止理由
    enum TCGStopReason tcg_reason;

    // Solve总时间
    double total_time;
  };

  struct Summary
  {
    Summary();
    
    std::string fullReport() const;

    bool isSolutionUsable() const;        

    std::string message = "ceres::Solve was not called.";

    TerminationType termination_type = FAILURE;

    std::vector<IterationSummary> iterations;    
  };

} // namespace manopt

#endif // MINIMIZER_H_
