#ifndef TRUST_REGION_H_
#define TRUST_REGION_H_
#include <iostream>
#include <cmath>
#include "common.hpp"
#include "manifold.hpp"
#include "minimizer.hpp" 
#include "tcg.hpp"

namespace manopt {

template <typename MType>
class TrustRegion {
  public:
    using MPoint  = typename MType::MPoint;
    using TVector = typename MType::TVector;
    using Scalar  = typename MType::Scalar; 
    using tCGResult = typename TruncatedConjugateGradient<MType>::Result;

    struct Options {
      int min_loop = 3;
      int max_loop = 60;
      int min_inner = 1;
      int max_inner = 1;
      double max_time = 20;
      double gradnorm_tol = 1e-5;
      double cost_change_tol = 1e-6;
      double kappa = 0.1;
      double theta = 1.0;
      double rho_prime = 0.1;
      double Delta_bar;       //初始化为 M::typicaldist()
      double Delta0;       
      bool use_rand = false;
      double rho_regularization = 1e3;
      bool verbosity = false;
      bool debug = false;

      Options(double db, int max_inner_) : 
        max_inner(max_inner_),
        Delta_bar(db),
        Delta0(db/8) { }
    };

    TrustRegion(typename Problem<MType>::Ptr problem_, double db=-1) :
      problem(problem_),
      options(db < 0 ? problem_->M()->typicaldist() : db, problem_->M()->dim()),
      tCG(problem_, options.theta, options.kappa, options.use_rand, 
          options.min_inner, options.max_inner, options.debug){
    }

    bool checkStoppingCriterionReached() {
      //检查梯度
      if (iteration_summary.step_is_accept && iteration_summary.gradient_norm < options.gradnorm_tol) {
        solver_summary->message = stringPrintf("Gradient tolerance reached. "
                                               "Gradient max norm: %lf <= %lf",
                                               iteration_summary.gradient_norm,
                                               options.gradnorm_tol);
        solver_summary->termination_type = CONVERGENCE;
        return true;
      }

      //检查cost_change
      if (iteration_summary.step_is_accept && 
          iteration_summary.cost_change < std::abs(iteration_summary.cost)*options.cost_change_tol) {
        solver_summary->message = stringPrintf("Cost change tolerance reached. "
                                               "Cost change: %lf <= %lf",
                                               iteration_summary.cost_change,
                                               std::abs(iteration_summary.cost)*options.cost_change_tol);
        solver_summary->termination_type = CONVERGENCE;
        return true;
      }      

      //检查执行时间
      const double total_solver_time = wallTimeInSeconds() - start_time_in_secs;
      if (total_solver_time > options.max_time) {
        solver_summary->message = stringPrintf("Maximum solver time reached. "
                                              "Total solver time: %lf >= %lf.",
                                              total_solver_time,
                                              options.max_time);
        solver_summary->termination_type = NO_CONVERGENCE;      
        return true;
      }

      //检查迭代次数
      if (iteration_summary.iteration >= options.max_loop) {
        solver_summary->message = stringPrintf("Maximum number of iterations reached. "
                                              "Number of iterations: %d.",
                                              iteration_summary.iteration);
        solver_summary->termination_type = NO_CONVERGENCE;

        return true;
      }

      return false;
    }

    void solve(MPoint& x, Summary* summary) {
      //记录开始时间
      start_time_in_secs = wallTimeInSeconds();
      solver_summary = summary;

      Scalar fx = problem->getCost(x);
      TVector fgradx = problem->getGradient(x);
      Scalar norm_grad = problem->M()->norm(x, fgradx);

      iteration_summary.cost = fx;
      //iteration_summary.cost_change = 0.0;
      iteration_summary.gradient_norm = norm_grad;
      iteration_summary.trust_region_radius = options.Delta0;
      iteration_summary.consecutive_TRplus = 0;
      iteration_summary.consecutive_TRminus = 0;

      solver_summary->iterations.clear();
      solver_summary->iterations.push_back(iteration_summary);


      //输出LOG
      if (options.verbosity) {
          std::string output = "iter acc/REJ   cost   |gradient| tr_ratio tr_radius tcg_iter tcg_time total_time tcg_reason\n";
          output += stringPrintf("% 4d        % 3.2e % 3.2e ",
              0,
              fx, 
              norm_grad
              );      
          std::cout << output << std::endl;
      }

      TVector eta;
      TVector Heta;

      while(!checkStoppingCriterionReached()) {

        double Delta = iteration_summary.trust_region_radius;

        //随机eta初始值
        if (options.use_rand) {
          eta = problem->M()->lincomb(x, 1e-6, problem->M()->randvec(x));
          while (problem->M()->norm(x, eta) > Delta) {
            eta = problem->M()->lincomb(x, 1.220703125e-4 /*sqrt(sqrt(eps))*/, eta);
          }
        } else {
          eta = problem->M()->zerovec(x);
        }

        double tCG_begin = wallTimeInSeconds();
        tCGResult res = tCG.solve(x, fgradx, eta, Delta);
        double tCG_elapsed = wallTimeInSeconds()-tCG_begin;
        eta = res.eta;
        Heta = res.Heta;

        iteration_summary.tcg_iterations = res.loopCount;
        iteration_summary.tcg_reason = res.stopReason;

        //检查eta和Heta跟柯西点结果比较~
        if (options.use_rand) {
          iteration_summary.used_cauchy = false;
          TVector Hg = problem->getHessian(x, fgradx);
          Scalar g_Hg = problem->M()->inner(x, fgradx, Hg);
          double tau_c;
          //g_Hg小于0，使用更小的stepsize。
          if (g_Hg <= 0) {
            tau_c = 1;
          } else {
            tau_c = std::min<Scalar>(std::pow(norm_grad, 3)/(Delta*g_Hg), 1);
          }

          TVector eta_c  = problem->M()->lincomb(x, -tau_c * Delta / norm_grad, fgradx);
          TVector Heta_c = problem->M()->lincomb(x, -tau_c * Delta / norm_grad, Hg);

          Scalar mdle  = fx + problem->M()->inner(x, fgradx, eta)   + 0.5*problem->M()->inner(x, Heta,   eta);
          Scalar mdlec = fx + problem->M()->inner(x, fgradx, eta_c) + 0.5*problem->M()->inner(x, Heta_c, eta_c);
          if (mdlec < mdle) {
            eta = eta_c;
            Heta = Heta_c;
            iteration_summary.used_cauchy = true;
            std::cout << "used_cauchy" << std::endl;
          }
        }        

        //下一个点x候选
        MPoint x_prop  = problem->M()->retr(x, eta);
        Scalar fx_prop = problem->getCost(x_prop);

        //计算信赖半径
        //rho的分子rhonum
        Scalar rhonum = fx - fx_prop;
        //方便计算rho的分母rhoden
        TVector vecrho = problem->M()->lincomb(x, 1, fgradx, .5, Heta);
        Scalar rhoden = -problem->M()->inner(x, eta, vecrho);

        //这个是在x在接近收敛时分子分母都比较小，所以会有扰动。
        //为了去掉扰动的影响都会加一个比较小的数
        Scalar rho_reg = std::max<Scalar>(1., std::abs(fx)) * 2.220446049250313e-16 /*eps*/ * options.rho_regularization;
        rhonum = rhonum + rho_reg;
        rhoden = rhoden + rho_reg;    

        if (options.debug > 0) {
          printf("DBG:     rhonum : % 3.6e\n", rhonum);
          printf("DBG:     rhoden : % 3.6e\n", rhoden);
        }   

        bool model_decreased = (rhoden >= 0); //TODO: 是否应该为>>> bool model_decreased = (rhonum >= 0)
        double rho = rhonum / rhoden;
        iteration_summary.trust_region_ratio = rho;

        if (options.debug > 0) {
          printf("DBG:   new f(x) : % 3.6e\n", fx_prop);
          printf("DBG:   used rho : % 3.6e\n", rho);
        }

        //根据rho调整信赖域
        if (rho < 0.25 || !model_decreased || std::isnan(rho)) {
          if (std::isnan(rho)) {
            std::cout << "WARNING: rho is nan" << std::endl;
          }
          iteration_summary.trust_region_radius /= 4;
          iteration_summary.consecutive_TRplus = 0;
          iteration_summary.consecutive_TRminus += 1;
        } else if (rho > 0.75 && (res.stopReason == NEGATIVE_CURVATURE || res.stopReason == EXCEEDED_TRUST_REGION)) {
          iteration_summary.trust_region_radius = std::min(2*iteration_summary.trust_region_radius, 
                                                           options.Delta_bar);
          iteration_summary.consecutive_TRplus += 1;
          iteration_summary.consecutive_TRminus = 0;
        } else {
          iteration_summary.consecutive_TRplus = 0;
          iteration_summary.consecutive_TRminus = 0;
        }

        //确定"接受"还是"拒绝"
        if (model_decreased && rho > options.rho_prime) {
          
          if (1) {
            problem->iterationCallback(x_prop);
            fx_prop = problem->getCost(x_prop);
          }

          //
          x = x_prop;
          fx = fx_prop;
          fgradx = problem->getGradient(x);
          norm_grad = problem->M()->norm(x, fgradx);

          iteration_summary.cost = fx;
          CHECK(rhonum >= 0);
          iteration_summary.cost_change = rhonum;
          iteration_summary.gradient_norm = norm_grad;
          iteration_summary.step_is_accept = true;
        } else {
          iteration_summary.cost_change = 0;
          iteration_summary.step_is_accept = false;
        }
            
        iteration_summary.tcg_time = tCG_elapsed;
        iteration_summary.total_time = wallTimeInSeconds() - start_time_in_secs;
        iteration_summary.iteration += 1;
        solver_summary->iterations.push_back(iteration_summary);

        if (options.verbosity) {
          std::string output= 
            stringPrintf("% 4d    %s % 3.2e % 3.2e % 3.2e % 3.2e % 4d    % 3.2e % 3.2e  %s",
              iteration_summary.iteration,
              (iteration_summary.step_is_accept ? "acc" : "REJ"), 
              iteration_summary.cost,
              iteration_summary.gradient_norm,
              iteration_summary.trust_region_ratio,
              iteration_summary.trust_region_radius,
              iteration_summary.tcg_iterations,
              iteration_summary.tcg_time,
              iteration_summary.total_time,
              TCGStopReasonStrings[iteration_summary.tcg_reason].c_str()
              );
          
          std::cout << output << std::endl;
        }
      }

    }

  private:
    IterationSummary iteration_summary;
    typename Problem<MType>::Ptr problem;
    Options options;
    TruncatedConjugateGradient<MType> tCG;

    Summary* solver_summary;
    double start_time_in_secs;

};

}  // namespace manopt


#endif  // TRUST_REGION_H_
