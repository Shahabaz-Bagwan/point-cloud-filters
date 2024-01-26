#pragma once
#include <vector>

namespace PCF {
  struct Point
  {
    double x, y, z;
    Point()                          = default;
    ~Point()                         = default;
    Point& operator=( const Point& ) = default;
    Point& operator=( Point&& )      = default;
    Point( const Point& p )          = default;
    Point( Point&& p ) noexcept      = default;

    Point( double x, double y, double z ) : x( x ), y( y ), z( z ) {}
  };

  using pointCloud = std::vector< Point >;

  struct NormalsAndCurvature
  {
    double nx, ny, nz, curvature;
    NormalsAndCurvature( double nx, double ny, double nz, double curvature )
      : nx( nx ), ny( ny ), nz( nz ), curvature( curvature )
    {
    }
    NormalsAndCurvature()                                        = default;
    NormalsAndCurvature( const NormalsAndCurvature& )            = default;
    NormalsAndCurvature( NormalsAndCurvature&& )                 = default;
    NormalsAndCurvature& operator=( NormalsAndCurvature&& )      = default;
    ~NormalsAndCurvature()                                       = default;
    NormalsAndCurvature& operator=( const NormalsAndCurvature& ) = default;
  };
} // namespace PCF
