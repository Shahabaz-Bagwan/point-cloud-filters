#pragma once

#include <flann/algorithms/dist.h>
#include <flann/flann.h>
#include <vector>

using FLANNIndex = ::flann::Index<::flann::L2_Simple< double > >;

struct Point
{
  double x, y, z;
};

using pointCloud = std::vector< Point >;

class KDtreeFlann
{
public:
  KDtreeFlann( pointCloud _input );

  KDtreeFlann();

  ~KDtreeFlann();

  int knnSearch( Point _point, size_t _k,
                 std::vector< size_t >& pointIdxKSearch,
                 std::vector< double >& pointKSquaredDistance );

  void setInputCloud( pointCloud _input );

  int radiusSearch( Point _point, double _radius,
                    std::vector< size_t >& indices_radius,
                    std::vector< double >& dists_radius );

private:
  std::vector< double > input, queryPt;

  size_t dim = 3, original_points;

  FLANNIndex* flann_index_;

  // Parameter for searching
  ::flann::SearchParams searchParam_;
};
