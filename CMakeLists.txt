cmake_minimum_required(VERSION 3.10)

project(manopt)

#set(CMAKE_BUILD_TYPE "RelWithDebInfo")
set(CMAKE_BUILD_TYPE "Release")
set (CMAKE_CXX_STANDARD 14)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# find_package(Eigen3 REQUIRED)
find_package(Glog REQUIRED)

include_directories(include/)
include_directories(/usr/include/eigen3/)

add_executable(manopt_test src/main.cc)
target_link_libraries(manopt_test glog)

# add_executable(manopt_tcg_test src/tcg_test.cc src/common.cc src/minimizer.cc)
add_executable(manopt_tr_test src/tr_test.cc src/tcg.cc src/common.cc src/minimizer.cc)
target_link_libraries(manopt_tr_test glog)

add_executable(rayleigh_quotient_test src/rayleigh_quotient_test.cc src/tcg.cc src/common.cc src/minimizer.cc)
target_link_libraries(rayleigh_quotient_test glog)

# add_executable(balm_test examples/balm.cc examples/PCRegistration/plane.cpp examples/PCRegistration/create_points.cpp examples/SE3/SO3.cpp examples/SE3/SE3.cpp examples/PCRegistration/arun.cpp examples/PCRegistration/plane_registration.cpp src/tcg.cc src/common.cc src/minimizer.cc)
# target_link_libraries(balm_test glog)
