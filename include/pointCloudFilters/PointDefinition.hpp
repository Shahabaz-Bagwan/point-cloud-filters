#pragma once
#include <vector>

namespace PCF {
  struct Point
  {
    double x, y, z;
  };

  using pointCloud = std::vector< Point >;

  struct normalsAndCurvature
  {
    double nx, ny, nz, curvature;
  };
} // namespace PCF
