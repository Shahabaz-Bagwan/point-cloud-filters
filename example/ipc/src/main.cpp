#include <iostream>
#include <pointCloudFilters/FileHandler.hpp>
#include <pointCloudFilters/IterativeClosestPoint.hpp>
#include <string>

int main( int argc, char* argv[] )
{
  using namespace PCF;
  if( argc < 2 ) {
    std::cout << "Please provide input file" << std::endl;
    return 1;
  }
  FileHandler fh;
  pointCloud A = fh.loadXYZfile( argv[ 1 ] );
  pointCloud B = fh.loadXYZfile( argv[ 2 ] );
  std::cout << "Input pointcloud size: " << A.size() << " " << B.size()
            << std::endl;

  IterativeClosestPoint ipc;
  pointCloud output;
  ipc.align( 100, 20, A, B, output );

  auto tx = ipc.getTxMatrix();

  for( const auto p : output )
    printf( "x:%f, y:%f\n", p.x, p.y );

  std::cout << tx << std::endl;
  fh.writeXYZfile( output, "op.xyz" );

  return 0;
}
