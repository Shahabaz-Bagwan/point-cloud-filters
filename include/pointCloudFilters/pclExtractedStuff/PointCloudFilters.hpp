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

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <KdtreeFlann.hpp>
#include <algorithm>
#include <cfloat>
#include <numeric>
#include <vector>

// Saves noramal and curvature values
struct normalsAndCurvature
{
  double nx, ny, nz, curvature;
};

class Filter3D
{

public:
  Filter3D();

  ~Filter3D();

  void statisticalOutlierRemoval( size_t _k, pointCloud input_,
                                  pointCloud& output );

  void voxelFilter( pointCloud input_, double _Delta, pointCloud& output );

  double getMean( size_t _k, pointCloud input_ );

  void MakeSmoothPointCloud( size_t _k, double smoothingFactor,
                             pointCloud input_, pointCloud& output );

  void euclideanClustering( double radius, size_t minClusterSize,
                            size_t maxClusterSize, pointCloud input_,
                            std::vector< std::vector< int > >& output );

  void euclideanClustering( double radius, size_t minClusterSize,
                            size_t maxClusterSize, pointCloud input_,
                            std::vector< pointCloud >& output );

  void
    findCentroidAndCovarianceMatrix( pointCloud cloud,
                                     Eigen::Matrix3f& covariance_matrix,
                                     Eigen::Matrix< double, 4, 1 >& centroid );

  void findNormals( pointCloud cloud, size_t numberOfneighbour,
                    std::vector< normalsAndCurvature >& output );

  void SetPasstroughLimits_X( double x_min, double x_max );

  void SetPasstroughLimits_Y( double y_min, double y_max );

  void SetPasstroughLimits_Z( double z_min, double z_max );

  void PassthroughFilter( pointCloud& input, pointCloud& output );

private:
  struct cloud_point_index_idx
  {
    size_t idx;
    size_t cloud_point_index;

    cloud_point_index_idx( size_t idx_, size_t cloud_point_index_ )
      : idx( idx_ ), cloud_point_index( cloud_point_index_ )
    {}
    bool operator<( const cloud_point_index_idx& p ) const
    {
      return ( idx < p.idx );
    }
  };

  std::pair< double, double > x_limits, y_limits, z_limits;
  bool x_passthough_limits_set, y_passthough_limits_set,
    z_passthough_limits_set;

  bool InRange( double x, double x1, double x2 )
  {
    return ( x >= x1 ) ? ( x <= x2 ) : false;
  }
};