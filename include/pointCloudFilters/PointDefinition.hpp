#pragma once
#include <vector>

struct Point
{
  double x, y, z;
};

using pointCloud = std::vector< Point >;

struct normalsAndCurvature
{
  double nx, ny, nz, curvature;
};