#include <algorithm>
#include <iterator>
#include <numeric>
#include <pointCloudFilters/IterativeClosestPoint.hpp>
#include <pointCloudFilters/KdtreeFlann.hpp>

namespace PCF {
  IterativeClosestPoint::IterativeClosestPoint()
    : correspondences_prev_mse_( std::numeric_limits< double >::max() )
  {
  }

  void IterativeClosestPoint::align( double maxDistance, size_t maxIterations,
                                     pointCloud& source, pointCloud target,
                                     pointCloud& output )
  {
    Eigen::Matrix4d transformationMatrix = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d finalTransformation  = Eigen::Matrix4d::Identity();

    size_t iteration = 0;

    correspondences_prev_mse_ = 0.0f;

    do {
      std::vector< Correspondence > correspondences;
      determineCorrespondences( correspondences, maxDistance, source, target );

      estimateRigidTransformation( source, target, correspondences,
                                   transformationMatrix );
      ++iteration;
      finalTransformation = transformationMatrix * finalTransformation;
      transformCloud( source, source, transformationMatrix );
      converged_ = hasConverged( iteration, maxIterations, correspondences );
    } while( !converged_ );
    transformCloud( source, output, transformationMatrix );
  }

  inline void IterativeClosestPoint::determineCorrespondences(
    std::vector< Correspondence >& correspondences, double maxDistance,
    pointCloud input, pointCloud target )
  {
    double maxDistSqr = maxDistance * maxDistance;

    correspondences.resize( input.size() );

    std::vector< long unsigned int > index( 1 );
    std::vector< double > distance( 1 );
    Correspondence corr;
    size_t nrValidCorrespondences = 0;
    KDtreeFlann tree( target );

    for( size_t i = 0; i < input.size(); i++ ) {
      tree.knnSearch( input[ i ], 1, index, distance );
      if( distance[ 0 ] > maxDistSqr ) {
        continue;
      }

      corr.indexQuery                             = i;
      corr.indexMatch                             = index[ 0 ];
      corr.distanceOrWeight                       = distance[ 0 ];
      correspondences[ nrValidCorrespondences++ ] = corr;
    }

    correspondences.resize( nrValidCorrespondences );
  }

  inline void IterativeClosestPoint::estimateRigidTransformation(
    pointCloud input, pointCloud target,
    std::vector< Correspondence > correspondences,
    Eigen::Matrix4d& transformationMatrix )
  {
    std::vector< unsigned long > sourceIt;
    std::vector< unsigned long > targetIt;

    for( auto& correspondence : correspondences ) {
      sourceIt.push_back( correspondence.indexQuery );
    }

    for( auto& correspondence : correspondences ) {
      targetIt.push_back( correspondence.indexMatch );
    }

    // Convert to Eigen format
    const int npts = static_cast< int >( sourceIt.size() );

    Eigen::Matrix< double, 3, Eigen::Dynamic > srcPts( 3, npts );
    Eigen::Matrix< double, 3, Eigen::Dynamic > tgt( 3, npts );

    Eigen::Matrix< double, 3, Eigen::Dynamic > cloudSrc( 3, npts );
    Eigen::Matrix< double, 3, Eigen::Dynamic > cloudTgt( 3, npts );

    int src  = 0;
    int trgt = 0;
    for( int i = 0; i < npts; ++i ) {
      cloudSrc( 0, i ) = input[ sourceIt[ src ] ].x;
      cloudSrc( 1, i ) = input[ sourceIt[ src ] ].y;
      cloudSrc( 2, i ) = input[ sourceIt[ src ] ].z;
      ++src;

      cloudTgt( 0, i ) = target[ targetIt[ trgt ] ].x;
      cloudTgt( 1, i ) = target[ targetIt[ trgt ] ].y;
      cloudTgt( 2, i ) = target[ targetIt[ trgt ] ].z;
      ++trgt;
    }

    // Call Umeyama directly from Eigen
    transformationMatrix = Eigen::umeyama( cloudSrc, cloudTgt, false );
    if( transformationMatrix != Eigen::Matrix4d::Identity() ) {
      transformation_matrix_ = transformation_matrix_ * transformationMatrix;
    }
  }

  inline void IterativeClosestPoint::transformCloud(
    pointCloud& input, pointCloud& output, const Eigen::Matrix4d& transform )
  {
    Eigen::Vector4d pt( 0.0f, 0.0f, 0.0f, 1.0f );
    Eigen::Vector4d ptT;
    Eigen::Matrix4d tr = transform.template cast< double >();
    pointCloud tmp;
    std::transform( input.begin(), input.end(), std::back_inserter( tmp ),
                    [ & ]( auto const& inputPnt ) {
                      pt[ 0 ] = inputPnt.x;
                      pt[ 1 ] = inputPnt.y;
                      pt[ 2 ] = inputPnt.z;

                      ptT = tr * pt;

                      return Point( ptT[ 0 ], ptT[ 1 ], ptT[ 2 ] );
                    } );
    output = std::move( tmp );
  }

  inline bool IterativeClosestPoint::hasConverged(
    size_t iterations, size_t maxIterations,
    std::vector< Correspondence > correspondences )
  {
    double correspondencesCurMse( std::numeric_limits< double >::max() );
    double rotationThreshold( 0.99999 );    // 0.256 degrees
    double translationThreshold( 0.0003f ); // 0.0003 meters
    double mseThresholdRelative(
      0.00001 ); // 0.001% of the previous MSE (relative error)
    double mseThresholdAbsolute( 1e-12 ); // MSE (absolute error)
    int iterationsSimilarTransforms( 0 );
    int maxIterationsSimilarTransforms( 1 );

    // Number of iterations has reached the maximum user imposed number of
    // iterations
    if( iterations <= maxIterations ) {
      correspondencesCurMse = 0;

      correspondencesCurMse =
        std::accumulate( correspondences.begin(), correspondences.end(),
                         double{}, []( double acc, Correspondence const& p ) {
                           return acc + p.distanceOrWeight;
                         } );

      correspondencesCurMse /= static_cast< double >( correspondences.size() );

      // Relative
      if( fabs( correspondencesCurMse - correspondences_prev_mse_ ) /
            correspondences_prev_mse_ <
          mseThresholdRelative ) {
        if( iterationsSimilarTransforms < maxIterationsSimilarTransforms ) {
          // Increment the number of transforms that the thresholds are allowed
          // to be similar
          ++iterationsSimilarTransforms;
          return ( false );
        }
        iterationsSimilarTransforms = 0;
        return ( true );
      }

      correspondences_prev_mse_ = correspondencesCurMse;
      return ( false );
    }
    return ( true );
  }
} // namespace PCF
