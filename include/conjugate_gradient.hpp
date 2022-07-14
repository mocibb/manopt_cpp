#ifndef CONJUGATE_GRADIENT_H_
#define CONJUGATE_GRADIENT_H_
#include <cmath>
#include <iostream>

#include "adaptive_linesearch.hpp"
#include "common.hpp"
#include "manifold.hpp"
#include "minimizer.hpp"

namespace manopt
{

  template <typename MType>
  class ConjugateGradient
  {
  public:
    using MPoint = typename MType::MPoint;
    using TVector = typename MType::TVector;
    using Scalar = typename MType::Scalar;

    enum BetaType
    {
      S_D,
      F_R,
      P_R,
      H_S,
      H_Z,
      L_S
    };

    struct Options
    {
      double minstepsize = 1e-10;
      int maxiter = 1000;
      double tolgradnorm = 1e-6;
      int storedepth = 20;
      BetaType beta_type = H_Z;
      double orth_value = 1e10;
      double gradnorm_tol = 1e-5;
      double cost_change_tol = 1e-6;
      int max_loop = 200;
      double max_time = 20;
    };

    ConjugateGradient(typename Problem<MType>::Ptr problem_)
        : problem(problem_), options() {}

    bool checkStoppingCriterionReached()
    {
      //检查梯度
      if (iteration_summary.gradient_norm <
          options.gradnorm_tol)
      {
        solver_summary->message = stringPrintf("Gradient tolerance reached. Gradient max norm: %lf <= %lf.",
                                               iteration_summary.gradient_norm,
                                               options.gradnorm_tol);
        solver_summary->termination_type = CONVERGENCE;
        return true;
      }

      //检查cost_change
      // if (iteration_summary.cost_change < options.cost_change_tol)
      // {
      //   solver_summary->message = stringPrintf("Cost change tolerance reached. Cost change: %lf <= %lf.",
      //                                          iteration_summary.cost_change,
      //                                          std::abs(iteration_summary.cost) * options.cost_change_tol);
      //   solver_summary->termination_type = CONVERGENCE;
      //   return true;
      // }

      //检查执行时间
      const double total_solver_time = wallTimeInSeconds() -
                                       start_time_in_secs;
      if (total_solver_time > options.max_time)
      {
        solver_summary->message = stringPrintf("Maximum solver time reached. Total solver time: %lf >= %lf.", total_solver_time,
                                               options.max_time);
        solver_summary->termination_type = NO_CONVERGENCE;
        return true;
      }

      //检查迭代次数
      if (iteration_summary.iteration >= options.max_loop)
      {
        solver_summary->message = stringPrintf("Maximum number of iterations reached. Number of iterations: %d.",
                                               iteration_summary.iteration);
        solver_summary->termination_type = NO_CONVERGENCE;

        return true;
      }

      return false;
    }

    TVector getPreconditioner(MPoint &x, const TVector &grad) { return grad; }

    void solve(MPoint &x, Summary *summary)
    {
      //记录开始时间
      start_time_in_secs = wallTimeInSeconds();
      solver_summary = summary;

      AdaptiveLineSearch<MType> linesearch(problem);

      Scalar cost = problem->getCost(x);
      TVector grad = problem->getGradient(x);

      Scalar norm_grad = problem->M()->norm(x, grad);

      TVector pgrad = getPreconditioner(x, grad);
      Scalar grad_pgrad = problem->M()->inner(x, grad, pgrad);
      TVector desc_dir = problem->M()->lincomb(x, -1, pgrad);

      int stepsize;
      MPoint new_x;

      iteration_summary.gradient_norm = norm_grad;
      while (!checkStoppingCriterionReached())
      {
        Scalar df0 = problem->M()->inner(x, grad, desc_dir);

        // std::cout << "df0 = " << df0 << std::endl;
        // std::cout << "iter = " << iteration_summary.iteration << ", cost = " << cost << ", gradnorm = " << norm_grad << std::endl;

        if (df0 > 0)
        {
          throw new std::runtime_error("got an ascent direction");
        }

        // std::cout << "x = " << x.transpose() << std::endl;
        // std::cout << "desc_dir = " << desc_dir.transpose() << std::endl;
        // std::cout << "cost = " << cost << std::endl;
        // std::cout << "df0 = " << df0 << std::endl;
        // std::cout << "=========" << std::endl;
        linesearch.search(x, desc_dir, cost, df0, stepsize, new_x);


        // std::cout << "stepsize = " << stepsize << std::endl;
        // std::cout << "new_x = " << new_x.transpose() << std::endl;

        Scalar new_cost = problem->getCost(new_x);
        TVector new_grad = problem->getGradient(new_x);

        Scalar new_norm_grad = problem->M()->norm(new_x, new_grad);
        TVector new_pgrad = getPreconditioner(new_x, new_grad);
        Scalar new_grad_pgrad = problem->M()->inner(new_x, new_grad, new_pgrad);

        // std::cout << "new_norm_grad = " << new_norm_grad << std::endl;
        // std::cout << "new_pgrad = " << new_pgrad.transpose() << std::endl;
        // std::cout << "new_grad_pgrad = " << new_grad_pgrad << std::endl;

        // if (i ++ > 10) {
        //   break;
        // }

        double beta;
        if (options.beta_type == S_D)
        {
          beta = 0;
          desc_dir = problem->M()->lincomb(new_x, -1, new_pgrad);
        }
        else
        {
          // std::cout << "x = " << x.transpose() << ", new_x = " << new_x.transpose() << std::endl;
          TVector old_grad = problem->M()->transp(x, new_x, grad);
          Scalar orth_grads =
              problem->M()->inner(new_x, old_grad, new_pgrad) / new_grad_pgrad;

          if (std::abs(orth_grads) >= options.orth_value)
          {
            beta = 0;
            desc_dir = problem->M()->lincomb(x, -1, new_pgrad);
          }
          else
          {
            desc_dir = problem->M()->transp(x, new_x, desc_dir);
            // std::cout << "desc_dir = " << desc_dir.transpose() << std::endl;

            switch (options.beta_type)
            {
            case F_R:
              beta = new_grad_pgrad / grad_pgrad;

              break;
            case P_R:
            {
              TVector diff =
                  problem->M()->lincomb(new_x, 1, new_grad, -1, old_grad);
              Scalar ip_diff = problem->M()->inner(new_x, new_pgrad, diff);
              beta = ip_diff / grad_pgrad;
              beta = std::max<Scalar>(0, beta);
            }
            break;

            case H_S:
            {
              TVector diff =
                  problem->M()->lincomb(new_x, 1, new_grad, -1, old_grad);
              Scalar ip_diff = problem->M()->inner(new_x, new_pgrad, diff);
              beta = ip_diff / problem->M()->inner(new_x, diff, desc_dir);
              // std::cout << "beta = " << beta << std::endl;
              beta = std::max<Scalar>(0, beta);
            }
            break;

            case H_Z:
            {
              TVector diff =
                  problem->M()->lincomb(new_x, 1, new_grad, -1, old_grad);
              TVector p_old_grad = problem->M()->transp(x, new_x, pgrad);
              TVector p_diff = problem->M()->lincomb(new_x, 1, new_pgrad, -1, p_old_grad);
              Scalar deno = problem->M()->inner(new_x, diff, desc_dir);
              Scalar numo = problem->M()->inner(new_x, diff, new_pgrad);
              numo = numo - 2 * problem->M()->inner(new_x, diff, p_diff) * problem->M()->inner(new_x, desc_dir, new_grad) / deno;
              beta = numo / deno;

              // Robustness(see Hager - Zhang paper mentioned above)
              Scalar desc_dir_norm = problem->M()->norm(new_x, desc_dir);
              Scalar eta_HZ = -1 / (desc_dir_norm * std::min<Scalar>(0.01, norm_grad));
              beta = std::max<Scalar>(beta, eta_HZ);
            }
            break;

            case L_S:
            {
              TVector diff =
                  problem->M()->lincomb(new_x, 1, new_grad, -1, old_grad);
              Scalar ip_diff = problem->M()->inner(new_x, new_pgrad, diff);
              Scalar denom = -problem->M()->inner(x, grad, desc_dir);
              Scalar betaLS = ip_diff / denom;
              Scalar betaCD = new_grad_pgrad / denom;
              beta = std::max<Scalar>(0, std::min<Scalar>(betaLS, betaCD));
            }

            break;
            default:
              throw new std::runtime_error("Unknown options.beta_type.");
            }

            desc_dir = problem->M()->lincomb(new_x, -1, new_pgrad, beta, desc_dir);
          }
        }

        iteration_summary.cost = new_cost;
        iteration_summary.cost_change = cost - new_cost;
        iteration_summary.gradient_norm = new_norm_grad;
        iteration_summary.iteration += 1;

        x = new_x;
        cost = new_cost;
        grad = new_grad;
        pgrad = new_pgrad;
        norm_grad = new_norm_grad;
        grad_pgrad = new_grad_pgrad;
      }
    }

  private:
    IterationSummary iteration_summary;
    typename Problem<MType>::Ptr problem;
    Options options;

    Summary *solver_summary;
    double start_time_in_secs;
  };

} // namespace manopt

#endif // TRUST_REGION_H_
