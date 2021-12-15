
#include <pointCloudFilters/IterativeClosestPoint.hpp>

IterativeClosestPoint::IterativeClosestPoint()
{

  correspondences_prev_mse_ = ( std::numeric_limits< double >::max() );
}

IterativeClosestPoint::~IterativeClosestPoint() {}

void IterativeClosestPoint::align( double max_distance, size_t maxIterations,
                                   pointCloud& source, pointCloud target,
                                   pointCloud& output )
{
  Eigen::Matrix4d transformation_matrix = Eigen::Matrix4d::Identity();
  Eigen::Matrix4d final_transformation_ = Eigen::Matrix4d::Identity();

  size_t iteration = 0;

  correspondences_prev_mse_ = 0.0f;

  do {
    std::vector< Correspondence > correspondences;
    determineCorrespondences( correspondences, max_distance, source, target );

    estimateRigidTransformation( source, target, correspondences,
                                 transformation_matrix );
    ++iteration;
    final_transformation_ = transformation_matrix * final_transformation_;
    transformCloud( source, source, transformation_matrix );
    converged = hasConverged( iteration, maxIterations, transformation_matrix,
                              correspondences );
  } while( !converged );
  transformCloud( source, output, transformation_matrix );
}

inline void IterativeClosestPoint::determineCorrespondences(
  std::vector< Correspondence >& correspondences, double max_distance,
  pointCloud input, pointCloud target )
{
  double max_dist_sqr = max_distance * max_distance;

  correspondences.resize( input.size() );

  std::vector< long unsigned int > index( 1 );
  std::vector< double > distance( 1 );
  Correspondence corr;
  size_t nr_valid_correspondences = 0;
  KDtreeFlann tree_( target );

  for( size_t i = 0; i < input.size(); i++ ) {
    tree_.knnSearch( input[ i ], 1, index, distance );
    if( distance[ 0 ] > max_dist_sqr )
      continue;

    corr.index_query                              = i;
    corr.index_match                              = index[ 0 ];
    corr.distance                                 = distance[ 0 ];
    correspondences[ nr_valid_correspondences++ ] = corr;
  }

  correspondences.resize( nr_valid_correspondences );
}

inline void IterativeClosestPoint::estimateRigidTransformation(
  pointCloud input, pointCloud target,
  std::vector< Correspondence > correspondences,
  Eigen::Matrix4d& transformation_matrix )
{
  std::vector< int > source_it;
  std::vector< int > target_it;

  for( auto indexIt = correspondences.begin(); indexIt != correspondences.end();
       ++indexIt )
    source_it.push_back( indexIt->index_query );

  for( auto indexIt = correspondences.begin(); indexIt != correspondences.end();
       ++indexIt )
    target_it.push_back( indexIt->index_match );

  // Convert to Eigen format
  const int npts = static_cast< int >( source_it.size() );

  Eigen::Matrix< double, 3, Eigen::Dynamic > cloud_src( 3, npts );
  Eigen::Matrix< double, 3, Eigen::Dynamic > cloud_tgt( 3, npts );
  int src = 0, trgt = 0;
  for( int i = 0; i < npts; ++i ) {
    cloud_src( 0, i ) = input[ source_it[ src ] ].x;
    cloud_src( 1, i ) = input[ source_it[ src ] ].y;
    cloud_src( 2, i ) = input[ source_it[ src ] ].z;
    ++src;

    cloud_tgt( 0, i ) = target[ target_it[ trgt ] ].x;
    cloud_tgt( 1, i ) = target[ target_it[ trgt ] ].y;
    cloud_tgt( 2, i ) = target[ target_it[ trgt ] ].z;
    ++trgt;
  }

  // Call Umeyama directly from Eigen
  transformation_matrix = Eigen::umeyama( cloud_src, cloud_tgt, false );
}

inline void
  IterativeClosestPoint::transformCloud( pointCloud& input, pointCloud& output,
                                         const Eigen::Matrix4d& transform )
{
  Eigen::Vector4d pt( 0.0f, 0.0f, 0.0f, 1.0f ), pt_t;
  Eigen::Matrix4d tr = transform.template cast< double >();
  output             = input;

  for( size_t i = 0; i < input.size(); ++i ) {
    pt[ 0 ] = input[ i ].x;
    pt[ 1 ] = input[ i ].y;
    pt[ 2 ] = input[ i ].z;

    pt_t = tr * pt;

    output[ i ].x = pt_t[ 0 ];
    output[ i ].y = pt_t[ 1 ];
    output[ i ].z = pt_t[ 2 ];
  }
}

inline bool IterativeClosestPoint::hasConverged(
  size_t iterations_, size_t max_iterations_, Eigen::Matrix4d& transformation_,
  std::vector< Correspondence > correspondences )
{
  double correspondences_cur_mse_( std::numeric_limits< double >::max() );
  double rotation_threshold_( 0.99999 );    // 0.256 degrees
  double translation_threshold_( 0.0003f ); // 0.0003 meters
  double mse_threshold_relative_(
    0.00001 ); // 0.001% of the previous MSE (relative error)
  double mse_threshold_absolute_( 1e-12 ); // MSE (absolute error)
  int iterations_similar_transforms_( 0 ),
    max_iterations_similar_transforms_( 1 );

  // Number of iterations has reached the maximum user imposed number of
  // iterations
  if( iterations_ <= max_iterations_ ) {
    correspondences_cur_mse_ = 0;

    for( size_t i = 0; i < correspondences.size(); ++i )
      correspondences_cur_mse_ += correspondences[ i ].distance;
    correspondences_cur_mse_ /= double( correspondences.size() );

    // Relative
    if( fabs( correspondences_cur_mse_ - correspondences_prev_mse_ ) /
          correspondences_prev_mse_ <
        mse_threshold_relative_ ) {
      if( iterations_similar_transforms_ <
          max_iterations_similar_transforms_ ) {
        // Increment the number of transforms that the thresholds are allowed to
        // be similar
        ++iterations_similar_transforms_;
        return ( false );
      } else {
        iterations_similar_transforms_ = 0;
        return ( true );
      }
    }

    correspondences_prev_mse_ = correspondences_cur_mse_;
    return ( false );
  }
  return ( true );
}