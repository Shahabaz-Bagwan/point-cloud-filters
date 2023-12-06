#include <iostream>
#include <ostream>
#include <pointCloudFilters/FileHandler.hpp>
#include <pointCloudFilters/IterativeClosestPoint.hpp>
#include <pointCloudFilters/KdtreeFlann.hpp>
#include <pointCloudFilters/PointCloudFilters.hpp>
#include <string>
#include <vector>

int main( int argc, char* argv[] )
{
  using namespace PCF;
  if( argc < 2 ) {
    std::cout << "Please provide input file" << std::endl;
    return 1;
  }
  FileHandler fh;
  pointCloud pc = fh.loadXYZfile( argv[ 1 ] );
  std::cout << "Input pointcloud size: " << pc.size() << std::endl;

  Filter3D filters;
  KDtreeFlann kdtree;
  IterativeClosestPoint ipc;

  pointCloud op;
  filters.voxelFilter( pc, std::stod( argv[ 2 ] ), op );
  std::cout << "Output pointcloud size: " << op.size() << std::endl;
  fh.writeXYZfile( op, "output.xyz" );

  return 0;
}
