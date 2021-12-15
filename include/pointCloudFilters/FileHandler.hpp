#include "KdtreeFlann.hpp"
#include <string>

class FileHandler
{
public:
  pointCloud loadXYZfile( const std::string& filename );

  int writeXYZfile( const pointCloud& pc, const std::string& filename );
};