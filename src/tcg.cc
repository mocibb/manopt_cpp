#include "tcg.hpp"

namespace manopt {

std::array<std::string, 6> TCGStopReasonStrings = {
  "negative curvature",
  "exceeded trust region",
  "reached target residual-kappa (linear)",
  "reached target residual-theta (superlinear)",
  "maximum inner iterations",
  "model increased"
};

}  // namespace manopt