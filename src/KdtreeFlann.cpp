#include <exception>
#include <pointCloudFilters/KdtreeFlann.hpp>
#include <stdexcept>

namespace PCF {
  KDtreeFlann::KDtreeFlann() : originalPoints_( 0 ) { queryPt_.resize( dim_ ); }

  void KDtreeFlann::setInputCloud( pointCloud input )
  {
    try {
      if( input.empty() ) {
        std::cerr << R"(Input size is "0" there will be error!)" << std::endl;
        throw std::logic_error{ R"(Input size is "0" there will be error!)" };
      }

      originalPoints_ = input.size();

      input_.resize( originalPoints_ * dim_ );

      searchParam_.max_neighbors = static_cast< int >( originalPoints_ );

      int c = 0;
      for( size_t i = 0; i < originalPoints_; i++ ) {
        input_[ c ]     = input[ i ].x;
        input_[ c + 1 ] = input[ i ].y;
        input_[ c + 2 ] = input[ i ].z;
        c += 3;
      }
      // build the indices tree
      flann_index_ = new FLANNIndex(
        ::flann::Matrix< double >( input_.data(), originalPoints_, dim_ ),
        ::flann::KDTreeSingleIndexParams( 15 ) ); // max 15 points/leaf
      flann_index_->buildIndex();

    } catch( int error ) {
      std::cout << "index out of bound error\n";
    }
  }

  KDtreeFlann::KDtreeFlann( pointCloud input )
  {
    try {
      if( input.empty() ) {
        std::cerr << R"(Input size is "0" there will be error!)" << std::endl;
        throw std::logic_error{ R"(Input size is "0" there will be error!)" };
      }
      originalPoints_ = input.size();

      dim_ = 3;

      input_.resize( originalPoints_ * dim_ );

      searchParam_.max_neighbors = static_cast< int >( originalPoints_ );

      queryPt_.resize( dim_ );

      int c = 0;
      for( int i = 0; i < originalPoints_; i++ ) {
        input_[ c ]     = input[ i ].x;
        input_[ c + 1 ] = input[ i ].y;
        input_[ c + 2 ] = input[ i ].z;
        c += 3;
      }

      // build the indices tree
      flann_index_ = new FLANNIndex(
        ::flann::Matrix< double >( input_.data(), originalPoints_, dim_ ),
        ::flann::KDTreeSingleIndexParams( 15 ) ); // max 15 points/leaf
      flann_index_->buildIndex();
    } catch( std::exception const& e ) {
      std::cout << "Error occurred " << e.what() << std::endl;
    }
  }

  // Provides points which has more than K neighbouring points
  int KDtreeFlann::knnSearch( Point point, size_t k,
                              std::vector< size_t >& pointIdxKSearch,
                              std::vector< double >& pointKSquaredDistance )
  {
    queryPt_[ 0 ] = point.x;
    queryPt_[ 1 ] = point.y;
    queryPt_[ 2 ] = point.z;
    pointIdxKSearch.resize( k );
    pointIdxKSearch.resize( k );

    ::flann::Matrix< size_t > indicesMatrix( pointIdxKSearch.data(), 1, k );
    ::flann::Matrix< double > distancesMatrix( pointKSquaredDistance.data(), 1,
                                               k );

    return ( flann_index_->knnSearch(
      ::flann::Matrix< double >( queryPt_.data(), 1, dim_ ), indicesMatrix,
      distancesMatrix, k, searchParam_ ) );
  }

  // Provides number of points in given radius
  int KDtreeFlann::radiusSearch( Point point, double radius,
                                 std::vector< size_t >& indicesRadius,
                                 std::vector< double >& distsRadius )
  {
    std::vector< std::vector< size_t > > indices( 1 );
    std::vector< std::vector< double > > dists( 1 );

    queryPt_[ 0 ] = point.x;
    queryPt_[ 1 ] = point.y;
    queryPt_[ 2 ] = point.z;

    int returnValue = ( flann_index_->radiusSearch(
      ::flann::Matrix< double >( queryPt_.data(), 1, dim_ ), indices, dists,
      static_cast< float >( radius * radius ), searchParam_ ) );

    indicesRadius = indices[ 0 ];
    distsRadius   = dists[ 0 ];

    return returnValue;
  }
} // namespace PCF
