#ifndef COST_FUNCTION_H_
#define COST_FUNCTION_H_

#include "manifold.hpp"

namespace manopt {

template <typename MType>
class GradientCostFunction {
  public:
    using Scalar  = typename MType::Scalar;
    using MPoint  = typename MType::MPoint;
    using TVector = typename MType::TVector;

    typedef std::shared_ptr<GradientCostFunction<MType>> Ptr;    
public:    
    virtual Scalar cost(const MPoint& x) const = 0;

    virtual TVector gradient(const MPoint& x) const = 0;

    virtual void iterationCallback(const MPoint& x){}    
};

}  // namespace manopt

#endif  // COST_FUNCTION_H_
