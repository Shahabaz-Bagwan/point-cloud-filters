#pragma once

#include "PointDefinition.hpp"
#include <flann/algorithms/dist.h>
#include <flann/flann.h>
#include <vector>

namespace PCF {
  using FLANNIndex = ::flann::Index< ::flann::L2_Simple< double > >;

  class KDtreeFlann
  {
  public:
    explicit KDtreeFlann( pointCloud input );

    KDtreeFlann();

    KDtreeFlann( const KDtreeFlann& )            = default;
    KDtreeFlann( KDtreeFlann&& )                 = delete;
    KDtreeFlann& operator=( const KDtreeFlann& ) = default;
    KDtreeFlann& operator=( KDtreeFlann&& )      = delete;
    ~KDtreeFlann()                               = default;

    int knnSearch( Point point, size_t k,
                   std::vector< size_t >& pointIdxKSearch,
                   std::vector< double >& pointKSquaredDistance );

    void setInputCloud( pointCloud input );

    int radiusSearch( Point point, double radius,
                      std::vector< size_t >& indicesRadius,
                      std::vector< double >& distsRadius );

  private:
    std::vector< double > input_, queryPt_;

    size_t dim_ = 3, originalPoints_;

    FLANNIndex* flann_index_;

    // Parameter for searching
    ::flann::SearchParams searchParam_;
  };

} // namespace PCF
