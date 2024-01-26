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

#include <math.h>

#include <algorithm>
#include <cfloat>
#include <numeric>
#include <pointCloudFilters/KdtreeFlann.hpp>
#include <pointCloudFilters/PointCloudFilters.hpp>
namespace PCF {

  Filter3D::Filter3D() = default;

  Filter3D::~Filter3D() = default;

  void Filter3D::statisticalOutlierRemoval( size_t k, pointCloud input,
                                            pointCloud& output )
  {
    KDtreeFlann tree;

    tree.setInputCloud( input );
    std::vector< size_t > indices;
    size_t size   = input.size();
    double stdMul = 1.0;

    std::vector< size_t > nnIndices( k );
    std::vector< double > nnDists( k );
    std::vector< double > distances;
    distances.reserve( size );
    indices.resize( size );

    // oii = output indices iterator, rii = removed indices iterator
    size_t oii = 0;
    size_t rii = 0;

    // First pass: Compute the mean distances for all points with respect to
    // their k nearest neighbours
    size_t validDistances = 0;
    for( size_t iii = 0; iii < size; ++iii ) // iii = input indices iterator
    {
      // Perform the nearest k search
      tree.knnSearch( input[ iii ], k, nnIndices, nnDists );

      // Calculate the mean distance to its neighbours
      double distSum = 0.0;
      for( int it = 1; it < k; ++it ) { // k = 0 is the query point
        distSum += sqrt( nnDists[ it ] );
      }
      distances[ iii ] = static_cast< double >( distSum / k );
      validDistances++;
    }

    // Estimate the mean and the standard deviation of the distance vector
    double sum   = 0;
    double sqSum = 0;

    sum   = std::accumulate( distances.begin(), distances.end(), 0 );
    sqSum = std::accumulate( distances.begin(), distances.end(), 0,
                             []( const double& lhs, const double& rhs ) {
                               return ( lhs + ( rhs * rhs ) );
                             } );

    double mean = sum / static_cast< double >( validDistances );
    double variance =
      ( sqSum - sum * sum / static_cast< double >( validDistances ) ) /
      ( static_cast< double >( validDistances ) - 1 );
    double stddev = std::sqrt( variance );

    double distanceThreshold = mean + stdMul * stddev;

    // Second pass: Classify the points on the computed distance threshold
    for( size_t iii = 0; iii < static_cast< int >( input.size() );
         ++iii ) // iii = input indices iterator
    {
      // Points having a too high average distance are outliers
      if( distances[ iii ] < distanceThreshold )
        // Otherwise it was a normal point for output (inlier)
        indices[ oii++ ] = iii;
    }

    // Resize the output arrays
    indices.resize( oii );
    for( size_t i = 0; i < indices.size(); i++ ) {
      output[ i ].x = input[ indices[ i ] ].x;
      output[ i ].y = input[ indices[ i ] ].y;
      output[ i ].z = input[ indices[ i ] ].z;
    }
    output.erase( output.begin() + indices.size(), output.end() );
    output.shrink_to_fit();
  }

  void Filter3D::voxelFilter( pointCloud input, double Delta,
                              pointCloud& output )
  {
    size_t size = input.size();
    output.resize( size );
    Eigen::Vector4i minB;
    Eigen::Vector4i maxB;
    Eigen::Vector4i divB;
    Eigen::Vector4i divbMul;
    Eigen::Array4d minP;
    Eigen::Array4d maxP;
    minP.setConstant( FLT_MAX );
    maxP.setConstant( -FLT_MAX );

    for( size_t i = 0; i < size; ++i ) {
      Eigen::Array4d pt;
      pt[ 0 ] = input[ i ].x;
      pt[ 1 ] = input[ i ].y;
      pt[ 2 ] = input[ i ].z;

      minP = minP.min( pt );
      maxP = maxP.max( pt );
    }

    for( size_t i = 0; i < size; i++ ) {
      double posX = 0.0f;
      double posY = 0.0f;
      double posZ = 0.0f;
      int absX    = 0;
      int absY    = 0;
      int absZ    = 0;

      // calculate the relative world co-ordinate position from minimum point
      posX = minP[ 0 ] - input[ i ].x;
      posY = minP[ 1 ] - input[ i ].y;
      posZ = minP[ 2 ] - input[ i ].z;

      /* calculate the number of grid required to reach the point from
      minimum point in respective axis direction*/
      absX = std::abs( std::ceil( posX / Delta ) );
      absY = std::abs( std::ceil( posY / Delta ) );
      absZ = std::abs( std::ceil( posZ / Delta ) );

      output[ i ].x = floor( absX * Delta - minP[ 0 ] );
      output[ i ].y = floor( absY * Delta - minP[ 1 ] );
      output[ i ].z = floor( absZ * Delta - minP[ 2 ] );
    }

    // Compute the minimum and maximum bounding box values
    minB[ 0 ] = static_cast< int >( floor( minP[ 0 ] / Delta ) );
    maxB[ 0 ] = static_cast< int >( floor( maxP[ 0 ] / Delta ) );
    minB[ 1 ] = static_cast< int >( floor( minP[ 1 ] / Delta ) );
    maxB[ 1 ] = static_cast< int >( floor( maxP[ 1 ] / Delta ) );
    minB[ 2 ] = static_cast< int >( floor( minP[ 2 ] / Delta ) );
    maxB[ 2 ] = static_cast< int >( floor( maxP[ 2 ] / Delta ) );

    // Compute the number of divisions needed along all axis
    divB      = maxB - minB + Eigen::Vector4i::Ones();
    divB[ 3 ] = 0;

    // Set up the division multiplier
    divbMul = Eigen::Vector4i( 1, divB[ 0 ], divB[ 0 ] * divB[ 1 ], 0 );

    // Storage for mapping leaf and pointcloud indexes
    std::vector< CloudPointIndexIdx > indexVector;
    indexVector.reserve( size );

    for( size_t it = 0; it < size; it++ ) {
      int ijk0 = static_cast< int >( floor( input[ it ].x / Delta ) -
                                     static_cast< double >( minB[ 0 ] ) );
      int ijk1 = static_cast< int >( floor( input[ it ].y / Delta ) -
                                     static_cast< double >( minB[ 1 ] ) );
      int ijk2 = static_cast< int >( floor( input[ it ].z / Delta ) -
                                     static_cast< double >( minB[ 2 ] ) );

      // Compute the centroid leaf index
      int idx = ijk0 * divbMul[ 0 ] + ijk1 * divbMul[ 1 ] + ijk2 * divbMul[ 2 ];
      indexVector.emplace_back( static_cast< size_t >( idx ), it );
    }
    auto compare = []( const CloudPointIndexIdx& lhs,
                       const CloudPointIndexIdx& rhs ) {
      return ( lhs.idx == rhs.idx );
    };

    auto predict = []( const CloudPointIndexIdx& lhs,
                       const CloudPointIndexIdx& rhs ) {
      return lhs.idx < rhs.idx;
    };

    std::sort( indexVector.begin(), indexVector.end(),
               std::less< CloudPointIndexIdx >() );

    // Third pass: count output cells
    // we need to skip all the same, adjacent idx values
    size_t total = 0;
    size_t index = 0;
    // first_and_last_indices_vector[i] represents the index in index_vector of
    // the first point in index_vector belonging to the voxel which corresponds
    // to the i-th output point, and of the first point not belonging to.
    std::vector< std::pair< size_t, size_t > > firstAndLastIndicesVector;
    // Worst case size
    firstAndLastIndicesVector.reserve( indexVector.size() );
    while( index < indexVector.size() ) {
      size_t i = index + 1;
      while( i < indexVector.size() &&
             indexVector[ i ].idx == indexVector[ index ].idx ) {
        ++i;
      }
      if( i - index >= 0 ) {
        ++total;
        firstAndLastIndicesVector.emplace_back( index, i );
      }
      index = i;
    }
    output.resize( total );

    index = 0;
    for( auto& cp : firstAndLastIndicesVector ) {
      // calculate centroid - sum values from all input points, that have the
      // same idx value in index_vector array
      size_t firstIndex = cp.first;
      size_t lastIndex  = cp.second;

      double x( 0.0f );
      double y( 0.0f );
      double z( 0.0f );

      // fill in the accumulator with leaf points
      size_t counter( 0 );
      for( size_t li = firstIndex; li < lastIndex; ++li ) {
        x = x + input[ indexVector[ li ].cloudPointIndex ].x;
        y = y + input[ indexVector[ li ].cloudPointIndex ].y;
        z = z + input[ indexVector[ li ].cloudPointIndex ].z;
        ++counter;
      }

      output[ index ].x = x / counter;
      output[ index ].y = y / counter;
      output[ index ].z = z / counter;

      ++index;
    }
    output.shrink_to_fit();
  }

  double Filter3D::getMean( size_t k, pointCloud input )
  {
    KDtreeFlann tree;

    tree.setInputCloud( input );
    std::vector< size_t > indices;
    size_t size   = input.size();
    double stdMul = 1.0;

    // The arrays to be used
    std::vector< size_t > nnIndices( k );
    std::vector< double > nnDists( k );
    std::vector< double > distances( size );
    indices.resize( size );

    // oii = output indices iterator, rii = removed indices iterator
    size_t oii = 0;
    size_t rii = 0;

    // First pass: Compute the mean distances for all points with respect to
    // their k nearest neighbours
    size_t validDistances = 0;
    for( size_t iii = 0; iii < size; ++iii ) // iii = input indices iterator
    {
      // Perform the nearest k search
      tree.knnSearch( input[ iii ], k, nnIndices, nnDists );

      // Calculate the mean distance to its neighbours
      double distSum = 0.0;
      for( size_t it = 1; it < k; ++it ) { // k = 0 is the query point
        distSum += sqrt( nnDists[ it ] );
      }
      distances[ iii ] = static_cast< double >( distSum / k );
      validDistances++;
    }

    // Estimate the mean and the standard deviation of the distance vector
    double sum   = 0;
    double sqSum = 0;
    sum          = std::accumulate( distances.begin(), distances.end(), 0 );
    sqSum        = std::accumulate( distances.begin(), distances.end(), 0,
                                    []( const double& lhs, const double& rhs ) {
                               return ( lhs + ( rhs * rhs ) );
                             } );

    double mean = sum / static_cast< double >( validDistances );
    double variance =
      ( sqSum - sum * sum / static_cast< double >( validDistances ) ) /
      ( static_cast< double >( validDistances ) - 1 );
    double stddev = sqrt( variance );
    return ( mean + stdMul * stddev );
  }

  void Filter3D::makeSmoothPointCloud( size_t k, double smoothingFactor,
                                       pointCloud input, pointCloud& output )
  {
    KDtreeFlann tree;

    tree.setInputCloud( input );
    std::vector< size_t > indices;
    size_t size   = input.size();
    double stdMul = 1.0;

    // The arrays to be used
    std::vector< double > avgX( size );
    std::vector< double > avgY( size );
    std::vector< double > avgZ( size );
    std::vector< size_t > nnIndices( k );
    std::vector< double > nnDists( k );
    std::vector< size_t > indicesRadius( 1 );
    std::vector< double > distRadius( 1 );
    std::vector< double > distances( size );
    indices.resize( size );

    // oii = output indices iterator, rii = removed indices iterator
    size_t oii = 0;
    size_t rii = 0;

    // First pass: Compute the mean distances for all points with respect to
    // their k nearest neighbours
    size_t validDistances = 0;
    for( size_t iii = 0; iii < size; ++iii ) // iii = input indices iterator
    {
      // Perform the nearest k search
      tree.knnSearch( input[ iii ], k, nnIndices, nnDists );

      // Calculate the mean distance to its neighbours
      double distSum = 0.0;
      for( int it = 1; it < k; ++it ) // k = 0 is the query point
        distSum += sqrt( nnDists[ it ] );
      distances[ iii ] = static_cast< double >( distSum / k );
      validDistances++;
    }

    // Estimate the mean and the standard deviation of the distance vector
    double sum    = 0;
    double sq_sum = 0;

    sum    = std::accumulate( distances.begin(), distances.end(), 0 );
    sq_sum = std::accumulate( distances.begin(), distances.end(), 0,
                              []( const double& lhs, const double& rhs ) {
                                return ( lhs + ( rhs * rhs ) );
                              } );

    double mean = sum / static_cast< double >( validDistances );

    double variance =
      ( sq_sum - sum * sum / static_cast< double >( validDistances ) ) /
      ( static_cast< double >( validDistances ) - 1 );

    double stddev = sqrt( variance );

    double radius = 0.0f;

    if( smoothingFactor < 1 ) {
      radius = ( mean + stdMul * stddev );
    } else {
      radius = smoothingFactor * ( mean + stdMul * stddev );
    }

    size_t counter = 0;
    for( size_t i = 0; i < size; i++ ) {
      if( tree.radiusSearch( input[ i ], radius, indicesRadius, distRadius ) >
          0 ) {
        size_t numberOfIndices = indicesRadius.size();
        if( numberOfIndices > 10 ) {
          for( size_t j = 0; j < numberOfIndices; j++ ) {
            // Accumulating the co-ordinates of the points which comes in the
            // volume sphere
            avgX[ counter ] = avgX[ counter ] + input[ indicesRadius[ j ] ].x;
            avgY[ counter ] = avgY[ counter ] + input[ indicesRadius[ j ] ].y;
            avgZ[ counter ] = avgZ[ counter ] + input[ indicesRadius[ j ] ].z;
          }
          // New point cloud with average points
          avgX[ counter ] = avgX[ counter ] / numberOfIndices;
          avgY[ counter ] = avgY[ counter ] / numberOfIndices;
          avgZ[ counter ] = avgZ[ counter ] / numberOfIndices;
          counter++;
        }
      }
    }

    // Resize again since that the removed points will change the size.
    avgX.resize( counter );
    avgY.resize( counter );
    avgZ.resize( counter );

    // Saving the reduced pointCloud in to new PCD file.
    output.resize( counter );

    for( int i = 0; i < counter; i++ ) {
      output[ i ].x = avgX[ i ];
      output[ i ].y = avgY[ i ];
      output[ i ].z = avgZ[ i ];
    }
  }

  void Filter3D::euclideanClustering( double radius, size_t minClusterSize,
                                      size_t maxClusterSize, pointCloud input,
                                      std::vector< pointCloud >& output )
  {
    KDtreeFlann tree;

    tree.setInputCloud( input );
    std::vector< bool > alreadyProcessed( input.size(), false );
    std::vector< size_t > radiusIndices;
    std::vector< double > radiusDistances;
    std::vector< std::vector< int > > clusters;
    size_t counter = 0;

    for( size_t i = 0; i < input.size(); i++ ) {
      if( alreadyProcessed[ i ] ) {
        continue;
      }

      std::vector< size_t > clusterPoints;
      size_t index = 0;
      clusterPoints.push_back( i );

      alreadyProcessed[ i ] = true;

      while( index < static_cast< int >( clusterPoints.size() ) ) {
        if( tree.radiusSearch( input[ clusterPoints[ index ] ], radius,
                               radiusIndices, radiusDistances ) == 0 ) {
          index++;
          continue;
        }

        for( unsigned long radiusIndice : radiusIndices ) {
          if( alreadyProcessed[ radiusIndice ] ) {
            continue;
          }

          clusterPoints.push_back( radiusIndice );
          alreadyProcessed[ radiusIndice ] = true;
        }
        index++;
      }

      if( clusterPoints.size() >= minClusterSize &&
          clusterPoints.size() <= maxClusterSize ) {
        std::vector< int > sortedCloud;
        pointCloud intermediatePointCloud;
        sortedCloud.resize( clusterPoints.size() );
        for( size_t j = 0; j < clusterPoints.size(); j++ ) {
          sortedCloud[ j ] = clusterPoints[ j ];
        }

        std::sort( sortedCloud.begin(), sortedCloud.end() );
        sortedCloud.erase(
          std::unique( sortedCloud.begin(), sortedCloud.end() ),
          sortedCloud.end() );
        intermediatePointCloud.resize( sortedCloud.size() );
        for( size_t k = 0; k < sortedCloud.size(); k++ ) {
          intermediatePointCloud[ k ].x = input[ sortedCloud[ k ] ].x;
          intermediatePointCloud[ k ].y = input[ sortedCloud[ k ] ].y;
          intermediatePointCloud[ k ].z = input[ sortedCloud[ k ] ].z;
        }
        output.push_back( intermediatePointCloud );
      }
    }
    sort( output.begin(), output.end(),
          []( const pointCloud& lhs, const pointCloud& rhs ) {
            return lhs.size() > rhs.size();
          } );
    output.shrink_to_fit();
  }

  void
    Filter3D::euclideanClustering( double radius, size_t minClusterSize,
                                   size_t maxClusterSize, pointCloud input,
                                   std::vector< std::vector< int > >& output )
  {
    KDtreeFlann tree;

    tree.setInputCloud( input );
    std::vector< bool > alreadyProceed( input.size(), false );
    std::vector< size_t > radiusIndices;
    std::vector< double > radiusDistances;
    std::vector< std::vector< int > > clusters;
    size_t counter = 0;

    for( size_t i = 0; i < input.size(); i++ ) {
      if( alreadyProceed[ i ] ) {
        continue;
      }

      std::vector< size_t > clusterPoints;
      size_t index = 0;
      clusterPoints.push_back( i );

      alreadyProceed[ i ] = true;

      while( index < static_cast< int >( clusterPoints.size() ) ) {
        if( tree.radiusSearch( input[ clusterPoints[ index ] ], radius,
                               radiusIndices, radiusDistances ) == 0 ) {
          index++;
          continue;
        }

        for( unsigned long radiusIndice : radiusIndices ) {
          if( alreadyProceed[ radiusIndice ] ) {
            continue;
          }

          clusterPoints.push_back( radiusIndice );
          alreadyProceed[ radiusIndice ] = true;
        }
        index++;
      }

      if( clusterPoints.size() >= minClusterSize &&
          clusterPoints.size() <= maxClusterSize ) {
        std::vector< int > sortedCloud;
        sortedCloud.resize( clusterPoints.size() );
        for( int j = 0; j < clusterPoints.size(); j++ ) {
          sortedCloud[ j ] = clusterPoints[ j ];
        }

        std::sort( sortedCloud.begin(), sortedCloud.end() );
        sortedCloud.erase(
          std::unique( sortedCloud.begin(), sortedCloud.end() ),
          sortedCloud.end() );
        output.push_back( sortedCloud );
      }
    }
    sort( output.begin(), output.end(),
          []( const std::vector< int >& lhs, const std::vector< int >& rhs ) {
            return lhs.size() > rhs.size();
          } );
    output.shrink_to_fit();
  }

  void Filter3D::findCentroidAndCovarianceMatrix(
    pointCloud cloud, Eigen::Matrix3f& covarianceMatrix,
    Eigen::Matrix< double, 4, 1 >& centroid )
  {
    // initialize matrix in row form to save the computations
    Eigen::Matrix< double, 1, 9, Eigen::RowMajor > accum =
      Eigen::Matrix< double, 1, 9, Eigen::RowMajor >::Zero();
    size_t pointCount = 0;
    pointCount        = cloud.size();
    // For each point in the cloud
    for( size_t i = 0; i < pointCount; ++i ) {
      accum[ 0 ] += cloud[ i ].x * cloud[ i ].x;
      accum[ 1 ] += cloud[ i ].x * cloud[ i ].y;
      accum[ 2 ] += cloud[ i ].x * cloud[ i ].z;
      accum[ 3 ] += cloud[ i ].y * cloud[ i ].y; // 4
      accum[ 4 ] += cloud[ i ].y * cloud[ i ].z; // 5
      accum[ 5 ] += cloud[ i ].z * cloud[ i ].z; // 8
      accum[ 6 ] += cloud[ i ].x;
      accum[ 7 ] += cloud[ i ].y;
      accum[ 8 ] += cloud[ i ].z;
    }

    accum /= static_cast< double >( pointCount );
    if( pointCount != 0 ) {
      centroid[ 0 ]                  = accum[ 6 ];
      centroid[ 1 ]                  = accum[ 7 ];
      centroid[ 2 ]                  = accum[ 8 ];
      centroid[ 3 ]                  = 1;
      covarianceMatrix.coeffRef( 0 ) = accum[ 0 ] - accum[ 6 ] * accum[ 6 ];
      covarianceMatrix.coeffRef( 1 ) = accum[ 1 ] - accum[ 6 ] * accum[ 7 ];
      covarianceMatrix.coeffRef( 2 ) = accum[ 2 ] - accum[ 6 ] * accum[ 8 ];
      covarianceMatrix.coeffRef( 4 ) = accum[ 3 ] - accum[ 7 ] * accum[ 7 ];
      covarianceMatrix.coeffRef( 5 ) = accum[ 4 ] - accum[ 7 ] * accum[ 8 ];
      covarianceMatrix.coeffRef( 8 ) = accum[ 5 ] - accum[ 8 ] * accum[ 8 ];
      covarianceMatrix.coeffRef( 3 ) = covarianceMatrix.coeff( 1 );
      covarianceMatrix.coeffRef( 6 ) = covarianceMatrix.coeff( 2 );
      covarianceMatrix.coeffRef( 7 ) = covarianceMatrix.coeff( 5 );
    }
  }

  void Filter3D::findNormals( pointCloud cloud, size_t numberOfneighbour,
                              std::vector< NormalsAndCurvature >& output )
  {
    Eigen::Matrix3f covarianceMatrix;
    Eigen::Matrix< double, 4, 1 > centroid;
    pointCloud processingCloud( numberOfneighbour );
    KDtreeFlann tree( cloud );
    std::vector< size_t > pointIdxKSearch( numberOfneighbour );
    std::vector< double > pointKSquaredDistance( numberOfneighbour );
    output.resize( cloud.size() );
    for( size_t i = 0; i < cloud.size(); i++ ) {
      tree.knnSearch( cloud[ i ], numberOfneighbour, pointIdxKSearch,
                      pointKSquaredDistance );
      for( size_t j = 0; j < numberOfneighbour; j++ ) {
        processingCloud[ j ].x = cloud[ pointIdxKSearch[ j ] ].x;
        processingCloud[ j ].y = cloud[ pointIdxKSearch[ j ] ].y;
        processingCloud[ j ].z = cloud[ pointIdxKSearch[ j ] ].z;
      }

      findCentroidAndCovarianceMatrix( processingCloud, covarianceMatrix,
                                       centroid );

      Eigen::Matrix3f::Scalar scale = covarianceMatrix.cwiseAbs().maxCoeff();
      Eigen::Matrix3f scaledCVM     = covarianceMatrix / scale;

      // Extract the smallest eigenvalue and its eigenvector
      Eigen::EigenSolver< Eigen::Matrix3f > es( scaledCVM );

      std::vector< double > eval( 3 );
      eval[ 0 ] = es.eigenvalues()[ 0 ].real();
      eval[ 1 ] = es.eigenvalues()[ 1 ].real();
      eval[ 2 ] = es.eigenvalues()[ 2 ].real();

      auto result = std::min_element( std::begin( eval ), std::end( eval ) );
      int minEvalIndex = std::distance( std::begin( eval ), result );

      Eigen::VectorXcf v = es.eigenvectors().col( minEvalIndex );
      output[ i ].nx     = v[ 0 ].real();
      output[ i ].ny     = v[ 1 ].real();
      output[ i ].nz     = v[ 2 ].real();
      if( ( ( output[ i ].nx * cloud[ i ].x ) +
            ( output[ i ].ny * cloud[ i ].y ) +
            ( output[ i ].nz * cloud[ i ].z ) ) > 0 ) {
        output[ i ].nx *= -1;
        output[ i ].ny *= -1;
        output[ i ].nz *= -1;
      }
      // Compute the curvature surface change
      double eigSum =
        scaledCVM.coeff( 0 ) + scaledCVM.coeff( 4 ) + scaledCVM.coeff( 8 );
      if( eigSum != 0 ) {
        output[ i ].curvature = fabsf( static_cast< float >(
          es.eigenvalues()[ minEvalIndex ].real() / eigSum ) );
      } else {
        output[ i ].curvature = 0;
      }
    }
  }

  void Filter3D::setPassthroughLimitsX( double xMin, double xMax )
  {
    if( xMin <= xMax ) {
      x_limits_.first           = xMin;
      x_limits_.second          = xMax;
      x_passthrough_limits_set_ = true;
    }
  }

  void Filter3D::setPassthroughLimitsY( double yMin, double yMax )
  {
    if( yMin <= yMax ) {
      y_limits_.first           = yMin;
      y_limits_.second          = yMax;
      y_passthrough_limits_set_ = true;
    }
  }

  void Filter3D::setPassthroughLimitsZ( double zMin, double zMax )
  {
    if( zMin <= zMax ) {
      z_limits_.first           = zMin;
      z_limits_.second          = zMax;
      z_passthrough_limits_set_ = true;
    }
  }

  void Filter3D::passthroughFilter( pointCloud& input, pointCloud& output )
  {
    pointCloud points = input;
    output.resize( points.size() );
    auto it = output.begin();

    if( x_passthrough_limits_set_ && y_passthrough_limits_set_ &&
        z_passthrough_limits_set_ ) {
      it = std::copy_if(
        points.begin(), points.end(), output.begin(),
        [ this ]( const Point& p ) {
          return ( inRange( p.x, x_limits_.first, x_limits_.second ) &&
                   inRange( p.y, y_limits_.first, y_limits_.second ) &&
                   inRange( p.z, z_limits_.first, z_limits_.second ) );
        } );
    } else if( x_passthrough_limits_set_ && y_passthrough_limits_set_ ) {
      it = std::copy_if(
        points.begin(), points.end(), output.begin(),
        [ this ]( const Point& p ) {
          return ( inRange( p.x, x_limits_.first, x_limits_.second ) &&
                   inRange( p.y, y_limits_.first, y_limits_.second ) );
        } );
    } else if( x_passthrough_limits_set_ && z_passthrough_limits_set_ ) {
      it = std::copy_if(
        points.begin(), points.end(), output.begin(),
        [ this ]( const Point& p ) {
          return ( inRange( p.x, x_limits_.first, x_limits_.second ) &&
                   inRange( p.z, z_limits_.first, z_limits_.second ) );
        } );
    } else if( y_passthrough_limits_set_ && z_passthrough_limits_set_ ) {
      it = std::copy_if(
        points.begin(), points.end(), output.begin(),
        [ this ]( const Point& p ) {
          return ( inRange( p.y, y_limits_.first, y_limits_.second ) &&
                   inRange( p.z, z_limits_.first, z_limits_.second ) );
        } );
    } else if( x_passthrough_limits_set_ ) {
      it = std::copy_if(
        points.begin(), points.end(), output.begin(),
        [ this ]( const Point& p ) {
          return ( inRange( p.x, x_limits_.first, x_limits_.second ) );
        } );
    } else if( y_passthrough_limits_set_ ) {
      it = std::copy_if(
        points.begin(), points.end(), output.begin(),
        [ this ]( const Point& p ) {
          return ( inRange( p.y, y_limits_.first, y_limits_.second ) );
        } );
    } else if( z_passthrough_limits_set_ ) {
      it = std::copy_if(
        points.begin(), points.end(), output.begin(),
        [ this ]( const Point& p ) {
          return ( inRange( p.z, z_limits_.first, z_limits_.second ) );
        } );
    }

    output.resize( std::distance( output.begin(), it ) );

    x_passthrough_limits_set_ = false;
    y_passthrough_limits_set_ = false;
    z_passthrough_limits_set_ = false;
  }

  bool Filter3D::inRange( double x, double x1, double x2 )
  {
    return ( x >= x1 ) ? ( x <= x2 ) : false;
  }
} // namespace PCF