/*
--------------------------------------------------------------------------
        contains 3D filter and tools
        1. Noise filter
        2. Voxel Filter
        3. Smoothing Filter
        4. Pass Through Filter
        5. Finding Mean of point cloud
        6. Euclidean Clustering
        7. Finding Normals of point cloud
        8. Find Centroid and Covariance Matrix
--------------------------------------------------------------------------
*/

#pragma once

#include "PointDefinition.hpp"
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <vector>

namespace PCF {

  class Filter3D
  {

  public:
    Filter3D();

    ~Filter3D();

    void statisticalOutlierRemoval( size_t k, pointCloud input,
                                    pointCloud& output );

    void voxelFilter( pointCloud input, double delta, pointCloud& output );

    double getMean( size_t k, pointCloud input );

    void makeSmoothPointCloud( size_t k, double smoothingFactor,
                               pointCloud input, pointCloud& output );

    void euclideanClustering( double radius, size_t minClusterSize,
                              size_t maxClusterSize, pointCloud input,
                              std::vector< std::vector< int > >& output );

    void euclideanClustering( double radius, size_t minClusterSize,
                              size_t maxClusterSize, pointCloud input,
                              std::vector< pointCloud >& output );

    void findCentroidAndCovarianceMatrix(
      pointCloud cloud, Eigen::Matrix3f& covarianceMatrix,
      Eigen::Matrix< double, 4, 1 >& centroid );

    void findNormals( pointCloud cloud, size_t numberOfneighbour,
                      std::vector< NormalsAndCurvature >& output );

    void setPassthroughLimitsX( double xMin, double xMax );

    void setPassthroughLimitsY( double yMin, double yMax );

    void setPassthroughLimitsZ( double zMin, double zMax );

    void passthroughFilter( pointCloud& input, pointCloud& output );

  private:
    struct CloudPointIndexIdx
    {
      size_t idx;
      size_t cloudPointIndex;

      CloudPointIndexIdx( const CloudPointIndexIdx& )            = default;
      CloudPointIndexIdx( CloudPointIndexIdx&& )                 = default;
      ~CloudPointIndexIdx()                                      = default;
      CloudPointIndexIdx()                                       = default;
      CloudPointIndexIdx& operator=( const CloudPointIndexIdx& ) = default;
      CloudPointIndexIdx& operator=( CloudPointIndexIdx&& )      = default;

      CloudPointIndexIdx( size_t idx, size_t cloudPointIndex )
        : idx( idx ), cloudPointIndex( cloudPointIndex )
      {
      }
      bool operator<( const CloudPointIndexIdx& p ) const
      {
        return ( idx < p.idx );
      }
    };

    std::pair< double, double > x_limits_, y_limits_, z_limits_;
    bool x_passthrough_limits_set_{ false }, y_passthrough_limits_set_{ false },
      z_passthrough_limits_set_{ false };

    static bool inRange( double x, double x1, double x2 );
  };
} // namespace PCF
