#ifndef ADAPTIVE_LINESEARCH_H_
#define ADAPTIVE_LINESEARCH_H_

#include "common.hpp"
#include "manifold.hpp"

namespace manopt
{

    template <typename MType>
    class AdaptiveLineSearch
    {
    public:
        using MPoint = typename MType::MPoint;
        using TVector = typename MType::TVector;
        using Scalar = typename MType::Scalar;

        struct Options
        {
            double ls_contraction_factor = .5;
            double ls_suff_decr = .5;
            int ls_max_steps = 10;
            int ls_initial_stepsize = 1;
        };

        AdaptiveLineSearch(typename Problem<MType>::Ptr problem_) : problem(problem_), init_alpha(-1)
        {
        }

        void search(const MPoint &x, const TVector &d, const Scalar &f0, const Scalar &df0, int &stepsize, MPoint &new_x)
        {
            double contraction_factor = options.ls_contraction_factor;
            double suff_decr = options.ls_suff_decr;
            int max_ls_steps = options.ls_max_steps;
            int initial_stepsize = options.ls_initial_stepsize;

            Scalar norm_d = problem->M()->norm(x, d);

            // 初始化alpha
            double alpha = initial_stepsize / norm_d;
            if (init_alpha > 0)
            {
                alpha = init_alpha;
            }

            TVector alpha_d = problem->M()->lincomb(x, alpha, d);
            MPoint newx = problem->M()->retr(x, alpha_d);

            Scalar newf = problem->getCost(newx);
            int cost_evaluations = 1;

            // 检查满足 Armijo 条件
            while (newf > f0 + suff_decr * alpha * df0)
            {
                alpha = contraction_factor * alpha;
                alpha_d = problem->M()->lincomb(x, alpha, d);
                newx = problem->M()->retr(x, alpha_d);
                newf = problem->getCost(newx);
                cost_evaluations++;

                // std::cout << "alpha = " << alpha << std::endl;

                if (cost_evaluations > max_ls_steps)
                {
                    break;
                }
            }

            // std::cout << "cost_evaluations = " << cost_evaluations << std::endl;
 
            if (newf > f0)
            {
                alpha = 0;
                newx = x;
            }

            stepsize = alpha * norm_d;
            new_x = newx;

            // 更新初始init_alpha
            if (cost_evaluations == 1)
            {
                init_alpha = 2 * alpha;
            }
            else if (cost_evaluations == 2)
            {
                init_alpha = alpha;
            }
            else
            {
                init_alpha = 2 * alpha;
            }
        }

    private:
        typename Problem<MType>::Ptr problem;
        Options options;
        double init_alpha;
    };

} // namespace manopt

#endif // ADAPTIVE_LINESEARCH_H_
