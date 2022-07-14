#include "tcg.hpp"
#include "common.hpp"
#include "minimizer.hpp"

namespace manopt {

Summary::Summary() {
}

std::string Summary::fullReport() const {
  std::string output;
  for (uint i=0; i<iterations.size(); i++) {
    auto& iteration_summary = iterations[i];
    if (i == 0){
      output = "iter acc/REJ     cost      |gradient|   tr_ratio tr_radius tcg_iter tcg_time total_time tcg_reason\n";
      output += stringPrintf("% 4d        % 3.6e % 3.4e \n",
          0,
          iteration_summary.cost,
          iteration_summary.gradient_norm
          );      
    } else {
        output+= 
          stringPrintf("% 4d    %s % 3.6e % 3.4e % 3.2e % 3.2e % 4d    % 3.2e % 3.2e  %s\n",
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
    }              
  }
  output += "\n"+message;
  return output;
}

bool Summary::isSolutionUsable() const {
  return termination_type == CONVERGENCE;
}

}  // namespace manopt