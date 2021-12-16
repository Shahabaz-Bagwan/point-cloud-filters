#include "PointDefinition.hpp"
#include <string>

namespace PCF {

  class FileHandler
  {
  public:
    pointCloud loadXYZfile( const std::string& filename );

    int writeXYZfile( const pointCloud& pc, const std::string& filename );
  };
} // namespace PCF