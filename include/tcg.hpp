#ifndef TCG_H_
#define TCG_H_
#include <iostream>
#include <array>
#include <cmath>
#include "manifold.hpp"
#include "problem.hpp"
#include <glog/logging.h>

namespace manopt {

enum TCGStopReason {
  NEGATIVE_CURVATURE,
  EXCEEDED_TRUST_REGION,
  LINEAR_CONVERGENCE,
  SUPERLINEAR_CONVERGENCE, 
  MAXIMUM_INNER_REACHED,
  MODEL_INCREASED
};

extern std::array<std::string, 6> TCGStopReasonStrings;

template <typename MType>
class TruncatedConjugateGradient {
  public:
    using MPoint  = typename MType::MPoint;
    using TVector = typename MType::TVector;
    using Scalar  = typename MType::Scalar;

    struct Result {
      TVector eta;
      TVector Heta;
      int loopCount;
      TCGStopReason stopReason;
    };

    TruncatedConjugateGradient(typename Problem<MType>::Ptr problem_,
                               double theta_, 
                               double kappa_,
                               bool use_rand_,
                               int min_loop_,
                               int max_loop_,
                               bool debug_) :
      problem(problem_),
      theta(theta_),
      kappa(kappa_),
      use_rand(use_rand_),
      min_loop(min_loop_),
      max_loop(max_loop_),
      debug(debug_) {
        CHECK(max_loop > min_loop);
      }

    // x:点所在流形上位置
    // grad: x处梯度
    // eta0: 是优化方向初值，非随机方法为zerovec(0)。
    // tr_radius: 信赖域半径
    // tCG求解子问题
    // ```math
    // \operatorname*{arg\,min}_{η ∈ T_xM}
    // m_x(η) \quad\text{where} 
    // m_x(η) = F(x) + ⟨\operatorname{grad}F(x),η⟩_x + \frac{1}{2}⟨\operatorname{Hess}F(x)[η],η⟩_x,
    // ```
    // 每次循环主要更新以下变量。
    // eta(优化变量的迭代值)
    // r(残差的迭代值)
    // z(残差preconditioner后的迭代值)
    // mdelta(每次优化方向)
    // 参考
    //https://manoptjl.org/v0.1/solvers/truncatedConjugateGradient/
    //
    //TODO加上preconditioner的实现
    Result solve(const MPoint& x, const TVector& grad, const TVector& eta0, double tr_radius) {
      //

      TVector eta = eta0, Heta, r, z;
      Scalar e_Pe, model_value;

      if (use_rand) {
        Heta = problem->getHessian(x, eta);
        r = problem->M()->lincomb(x, 1, grad, 1, Heta); 
        e_Pe = problem->M()->inner(x, eta, eta);
        z = r;
        model_value = problem->M()->inner(x, eta, grad) + 0.5 * problem->M()->inner(x, eta, Heta);
      } else {
        Heta = problem->M()->zerovec(x);
        r = grad;
        e_Pe = 0;
        z = r; /* get_preconditioner(p, x, r) not support yet */
        model_value = 0;
      }

      //初始化残差
      Scalar r_r = problem->M()->inner(x, r, r);
      Scalar norm_r = sqrt(r_r);
      Scalar norm_r0 = norm_r;

      Scalar z_r = problem->M()->inner(x, z, r);
      Scalar d_Pd = z_r;
      TVector mdelta = z;
      Scalar e_Pd = use_rand ? -problem->M()->inner(x, eta, mdelta) : 0;

      for (int j=0; j<max_loop; j++) {

        TVector Hmdelta = problem->getHessian(x, mdelta);
        Scalar d_Hd = problem->M()->inner(x, mdelta, Hmdelta);
        Scalar alpha = z_r / d_Hd;

        //e_Pe_new为了方便计算|eta|^2_P。
        Scalar e_Pe_new = e_Pe + 2 * alpha * e_Pd + alpha * alpha * d_Pd;

#if 0
        std::cout << "e_Pe_new = "  << e_Pe_new
                  << ",\t|eta|^2 = " <<  problem->M()->inner(x,
                                              problem->M()->lincomb(x, 1, eta, -alpha, mdelta),
                                              problem->M()->lincomb(x, 1, eta, -alpha, mdelta)) 
                  << std::endl;
#endif                  
        
        if (debug) {
          printf("DBG:   (r,r)  : % 3.6e\n", r_r);
          printf("DBG:   (d,Hd) : % 3.6e\n", d_Hd);
          printf("DBG:   alpha  : % 3.6e\n", alpha);
        }
        
        //如果极小值点处的d_Hd应该>=0，
        //e_Pe_new >= tr_radius*tr_radius表示超出指定的信赖域半径
        if (d_Hd <= 0 || e_Pe_new >= tr_radius*tr_radius) {
          //求解tau满足 | eta+ tau*mdelta |_P = Delta，这样返回的|eta|正好为信赖域半径
          Scalar tau = (-e_Pd + sqrt(e_Pd*e_Pd + d_Pd * (tr_radius*tr_radius - e_Pe))) / d_Pd;
          if (debug) {
            printf("DBG:     tau  : % 3.6e\n", tau);
          }
          eta  = problem->M()->lincomb(x, 1, eta, -tau, mdelta);
          Heta  = problem->M()->lincomb(x, 1, Heta, -tau, Hmdelta);

          if (d_Hd <= 0) {
            return Result{eta, Heta, j+1, NEGATIVE_CURVATURE};
          } else {
            return Result{eta, Heta, j+1, EXCEEDED_TRUST_REGION};
          }
        }

        e_Pe = e_Pe_new;

        TVector new_eta = problem->M()->lincomb(x, 1, eta, -alpha, mdelta);        
        //应该是problem->getHessian(x, new_eta)
        //TODO 检查new_Hη是否能很好近似 
        TVector new_Heta = problem->M()->lincomb(x, 1, Heta, -alpha, Hmdelta);
        
        // No negative curvature and eta - alpha * (mdelta) inside TR: accept it.
        Scalar new_model_value = problem->M()->inner(x, new_eta, grad) 
                         + 0.5 * problem->M()->inner(x, new_eta, new_Heta);
        if (new_model_value > model_value) {
          return Result{eta, Heta, j+1, MODEL_INCREASED};
        }

        eta = new_eta;
        Heta = new_Heta;
        model_value = new_model_value;
        r = problem->M()->lincomb(x, 1, r, -alpha, Hmdelta);
        
        r_r = problem->M()->inner(x, r, r);
        norm_r = sqrt(r_r);

        //如果norm_r已经很小了
        if (j+1 >= min_loop 
          && norm_r <= norm_r0*std::min(std::pow(norm_r0, theta), kappa)) {  
          if (kappa < std::pow(norm_r0, theta)) {
            return Result{eta, Heta, j+1, LINEAR_CONVERGENCE};       //linear convergence
          } else {
            return Result{eta, Heta, j+1, SUPERLINEAR_CONVERGENCE};  //superlinear convergence
          }
        }

        // Precondition the r.
        z = use_rand ? r : r /* get_preconditioner(p, x, r) not support yet */;
        Scalar zold_rold = z_r;
        z_r = problem->M()->inner(x, z, r);
        //# Compute new search direction.
        Scalar beta = z_r / zold_rold;
        mdelta = problem->M()->lincomb(x, 1, z, beta, mdelta);
        //投影δ，由于存在误差可能会偏离切平面
        mdelta = problem->M()->tangent(x, mdelta);
        //方便计算是否超出信赖域
        e_Pd = beta * (alpha * d_Pd + e_Pd);
        d_Pd = z_r + beta * beta * d_Pd;
      }
      
      return Result{eta, Heta, max_loop, MAXIMUM_INNER_REACHED};
    }

  private:
    typename Problem<MType>::Ptr problem;

    double theta;
    double kappa;
    bool use_rand;
    int min_loop;
    int max_loop;    
    bool debug;
};

}  // namespace manopt


#endif  // TCG
