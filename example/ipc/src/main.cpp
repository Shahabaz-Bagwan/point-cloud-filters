#include <chrono>
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
  auto start = std::chrono::high_resolution_clock::now();
  ipc.align( 100, 20, A, B, output );
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration =
    std::chrono::duration_cast< std::chrono::microseconds >( stop - start );
  std::cout << "time: " << duration.count() << "µs" << std::endl;

  // for( const auto p : output ) {
  //   printf( "x:%f, y:%f\n", p.x, p.y );
  // }

  auto tx = ipc.getTxMatrix();
  std::cout << tx << std::endl;
  fh.writeXYZfile( output, "op.xyz" );

  return 0;
}
