#include "rotation.hpp"
#include "euclidean.hpp"
#include "product_manifold.hpp"
#include <iostream>
#include <Eigen/Core>

using namespace manopt;


int main() {
  int k = 3;
  RotationX<double, 3> R(k);
  EuclideanX<double, 3> t(k);
  ProductManifold<decltype(R), decltype(t)> M(R, t);

  using MType = decltype(M);
  MType::MPoint x = M.rand();
  std::cout << x.first << std::endl;
  std::cout << x.second << std::endl;

  return 0;
}