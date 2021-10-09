#include <KdtreeFlann.hpp>

KDtreeFlann::KDtreeFlann() { queryPt.resize( dim ); }

KDtreeFlann::~KDtreeFlann() {}

void KDtreeFlann::setInputCloud( pointCloud _input )
{
  try {
    if( _input.size() == 0 ) {
      std::cerr << "Input size is \"0\" there will be error!\n";
      throw -1;
    }

    original_points = _input.size();

    input.resize( original_points * dim );

    searchParam_.max_neighbors = original_points;

    int c = 0;
    for( size_t i = 0; i < original_points; i++ ) {
      input[ c ]     = _input[ i ].x;
      input[ c + 1 ] = _input[ i ].y;
      input[ c + 2 ] = _input[ i ].z;
      c += 3;
    }
    // build the indices tree
    flann_index_ = new FLANNIndex(
      ::flann::Matrix< double >( ( &input[ 0 ] ), original_points, dim ),
      ::flann::KDTreeSingleIndexParams( 15 ) ); // max 15 points/leaf
    flann_index_->buildIndex();

  } catch( int error ) {
    std::cout << "index out of bound error\n";
  }
}

KDtreeFlann::KDtreeFlann( pointCloud _input )
{
  try {
    if( _input.size() == 0 ) {
      std::cerr << "Input size is \"0\" there will be error!\n";
      throw -1;
    }
    original_points = _input.size();

    dim = 3;

    input.resize( original_points * dim );

    searchParam_.max_neighbors = original_points;

    queryPt.resize( dim );

    int c = 0;
    for( int i = 0; i < original_points; i++ ) {
      input[ c ]     = _input[ i ].x;
      input[ c + 1 ] = _input[ i ].y;
      input[ c + 2 ] = _input[ i ].z;
      c += 3;
    }

    // build the indices tree
    flann_index_ = new FLANNIndex(
      ::flann::Matrix< double >( ( &input[ 0 ] ), original_points, dim ),
      ::flann::KDTreeSingleIndexParams( 15 ) ); // max 15 points/leaf
    flann_index_->buildIndex();
  } catch( int error ) {
    std::cout << "index out of bound error\n";
  }
}

// Provides points which has more than K neighbouring points
int KDtreeFlann::knnSearch( Point _point, size_t _k,
                            std::vector< size_t >& pointIdxKSearch,
                            std::vector< double >& pointKSquaredDistance )
{
  queryPt[ 0 ] = _point.x;
  queryPt[ 1 ] = _point.y;
  queryPt[ 2 ] = _point.z;
  pointIdxKSearch.resize( _k );
  pointIdxKSearch.resize( _k );

  ::flann::Matrix< size_t > indices_matrix( &pointIdxKSearch[ 0 ], 1, _k );
  ::flann::Matrix< double > distances_matrix( &pointKSquaredDistance[ 0 ], 1,
                                              _k );

  return ( flann_index_->knnSearch(
    ::flann::Matrix< double >( &queryPt[ 0 ], 1, dim ), indices_matrix,
    distances_matrix, _k, searchParam_ ) );
}

// Provides number of points in given radius
int KDtreeFlann::radiusSearch( Point _point, double _radius,
                               std::vector< size_t >& indices_radius,
                               std::vector< double >& dists_radius )
{
  std::vector< std::vector< size_t > > indices( 1 );
  std::vector< std::vector< double > > dists( 1 );

  queryPt[ 0 ] = _point.x;
  queryPt[ 1 ] = _point.y;
  queryPt[ 2 ] = _point.z;

  int return_value = ( flann_index_->radiusSearch(
    ::flann::Matrix< double >( &queryPt[ 0 ], 1, dim ), indices, dists,
    _radius * _radius, searchParam_ ) );

  indices_radius = indices[ 0 ];
  dists_radius   = dists[ 0 ];

  return return_value;
}