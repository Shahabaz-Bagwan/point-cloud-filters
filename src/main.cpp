#include <IterativeClosestPoint.hpp>
#include <KdtreeFlann.hpp>
#include <PointCloudFilters.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

pointCloud loadXYZfile(const std::string filename) {
  std::fstream file;
  file.open(filename);
  pointCloud pc;
  Point p;
  std::string x, y, z;

  if (!file.is_open())
    std::cout << "can't open input file" << std::endl;

  while (true) {
    if (std::getline(file, x, ' ') && std::getline(file, y, ' ') &&
        std::getline(file, z)) {
      p.x = std::stod(x);
      p.y = std::stod(y);
      p.z = std::stod(z);
      pc.push_back(p);
    }
    if (file.eof())
      break;
  }
  return pc;
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cout << "Please provide input file" << std::endl;
    return 1;
  }
  pointCloud pc = loadXYZfile(argv[1]);
  std::cout << "point cloud size: " << pc.size() << std::endl;
  return 0;
}
