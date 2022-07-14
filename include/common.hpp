#ifndef COMMON_H_
#define COMMON_H_

#include <sys/time.h>
#include <string>
#include <cstdio>

namespace manopt {

double wallTimeInSeconds();

std::string stringPrintf(const char* format, ...);

}  // namespace manopt

#endif  // COMMON_H_
