#include <fstream>
#include <iostream>
#include <pointCloudFilters/FileHandler.hpp>

namespace PCF {

  pointCloud FileHandler::loadXYZfile( const std::string& filename )
  {
    std::ifstream file;
    file.open( filename );
    pointCloud pc;
    Point p;
    std::string x, y, z;

    if( !file.is_open() )
      std::cout << "can't open input file" << std::endl;

    while( true ) {
      if( std::getline( file, x, ' ' ) && std::getline( file, y, ' ' ) &&
          std::getline( file, z ) ) {
        p.x = std::stod( x );
        p.y = std::stod( y );
        p.z = std::stod( z );
        pc.push_back( p );
      }
      if( file.eof() )
        break;
    }
    file.close();
    return pc;
  }

  int FileHandler::writeXYZfile( const pointCloud& pc,
                                 const std::string& filename )
  {
    std::ofstream xyzFile;
    xyzFile.open( filename );

    if( !xyzFile.is_open() ) {
      std::cout << "can't open file for writing" << std::endl;
      return -1;
    }

    for( size_t i = 0; i < ( pc.size() - 1 ); i++ ) {
      xyzFile << pc[ i ].x << " " << pc[ i ].y << " " << pc[ i ].z << std::endl;
    }
    xyzFile << pc[ pc.size() - 1 ].x << " " << pc[ pc.size() - 1 ].y << " "
            << pc[ pc.size() - 1 ].z;
    xyzFile.close();
    return 0;
  }
} // namespace PCF
