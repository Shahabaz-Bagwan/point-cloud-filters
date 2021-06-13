/*
--------------------------------------------------------------------------
        contains 3D filter and tools
        1. Noise filter
        2. Voxel Filter
        3. Smoothing Filter
        4. Pass Through Filter
        5. Finding Mean of point cloud
        6. Euclidean Clustering
        7. Finding Normals of point cloud
        8. Find Centroid and Covariance Matrix
--------------------------------------------------------------------------
*/

#include <PointCloudFilters.hpp>

Filter3D::Filter3D() {
  x_passthough_limits_set = false;
  y_passthough_limits_set = false;
  z_passthough_limits_set = false;
}

Filter3D::~Filter3D() {}

void Filter3D::statisticalOutlierRemoval(size_t _k, pointCloud input_,
                                         pointCloud &output) {
  KDtreeFlann tree;

  tree.setInputCloud(input_);
  std::vector<size_t> indices;
  size_t size = input_.size();
  double std_mul_ = 1.0;

  std::vector<size_t> nn_indices(_k);
  std::vector<double> nn_dists(_k);
  std::vector<double> distances(size);
  indices.resize(size);
  size_t oii = 0,
         rii =
             0; // oii = output indices iterator, rii = removed indices iterator

  // First pass: Compute the mean distances for all points with respect to their
  // k nearest neighbours
  size_t valid_distances = 0;
  for (size_t iii = 0; iii < size; ++iii) // iii = input indices iterator
  {
    // Perform the nearest k search
    tree.knnSearch(input_[iii], _k, nn_indices, nn_dists);

    // Calculate the mean distance to its neighbours
    double dist_sum = 0.0;
    for (int it = 1; it < _k; ++it) // k = 0 is the query point
      dist_sum += sqrt(nn_dists[it]);
    distances[iii] = static_cast<double>(dist_sum / _k);
    valid_distances++;
  }

  // Estimate the mean and the standard deviation of the distance vector
  double sum = 0, sq_sum = 0;

  sum = std::accumulate(distances.begin(), distances.end(), 0);
  sq_sum = std::accumulate(distances.begin(), distances.end(), 0,
                           [](double &lhs, double &rhs) {
                             return static_cast<double>(lhs + (rhs * rhs));
                           });

  double mean = sum / static_cast<double>(valid_distances);
  double variance =
      (sq_sum - sum * sum / static_cast<double>(valid_distances)) /
      (static_cast<double>(valid_distances) - 1);
  double stddev = std::sqrt(variance);

  double distance_threshold = mean + std_mul_ * stddev;

  // Second pass: Classify the points on the computed distance threshold
  for (size_t iii = 0; iii < static_cast<int>(input_.size());
       ++iii) // iii = input indices iterator
  {
    // Points having a too high average distance are outliers
    if (distances[iii] < distance_threshold)
      // Otherwise it was a normal point for output (inlier)
      indices[oii++] = iii;
  }

  // Resize the output arrays
  indices.resize(oii);
  for (size_t i = 0; i < indices.size(); i++) {
    output[i].x = input_[indices[i]].x;
    output[i].y = input_[indices[i]].y;
    output[i].z = input_[indices[i]].z;
  }
  output.erase(output.begin() + indices.size(), output.end());
  output.shrink_to_fit();
}

void Filter3D::voxelFilter(pointCloud input_, double _Delta,
                           pointCloud &output) {
  size_t size = input_.size();
  output.resize(size);
  Eigen::Vector4i min_b_, max_b_, div_b_, divb_mul_;
  Eigen::Array4d min_p, max_p;
  min_p.setConstant(FLT_MAX);
  max_p.setConstant(-FLT_MAX);

  for (size_t i = 0; i < size; ++i) {
    Eigen::Array4d pt;
    pt[0] = input_[i].x;
    pt[1] = input_[i].y;
    pt[2] = input_[i].z;

    min_p = min_p.min(pt);
    max_p = max_p.max(pt);
  }

  for (size_t i = 0; i < size; i++) {
    double pos_x, pos_y, pos_z;
    int abs_x, abs_y, abs_z;

    // calculate the relative world co-ordinate position from minimum point
    pos_x = min_p[0] - input_[i].x;
    pos_y = min_p[1] - input_[i].y;
    pos_z = min_p[2] - input_[i].z;

    /* calculate the number of grid required to reach the point from
    minimum point in respective axis direction*/
    abs_x = std::abs(std::ceil(pos_x / _Delta));
    abs_y = std::abs(std::ceil(pos_y / _Delta));
    abs_z = std::abs(std::ceil(pos_z / _Delta));

    output[i].x = floor(abs_x * _Delta - min_p[0]);
    output[i].y = floor(abs_y * _Delta - min_p[1]);
    output[i].z = floor(abs_z * _Delta - min_p[2]);
  }

  // Compute the minimum and maximum bounding box values
  min_b_[0] = static_cast<int>(floor(min_p[0] / _Delta));
  max_b_[0] = static_cast<int>(floor(max_p[0] / _Delta));
  min_b_[1] = static_cast<int>(floor(min_p[1] / _Delta));
  max_b_[1] = static_cast<int>(floor(max_p[1] / _Delta));
  min_b_[2] = static_cast<int>(floor(min_p[2] / _Delta));
  max_b_[2] = static_cast<int>(floor(max_p[2] / _Delta));

  // Compute the number of divisions needed along all axis
  div_b_ = max_b_ - min_b_ + Eigen::Vector4i::Ones();
  div_b_[3] = 0;

  // Set up the division multiplier
  divb_mul_ = Eigen::Vector4i(1, div_b_[0], div_b_[0] * div_b_[1], 0);

  // Storage for mapping leaf and pointcloud indexes
  std::vector<cloud_point_index_idx> index_vector;
  index_vector.reserve(size);

  for (size_t it = 0; it < size; it++) {
    int ijk0 = static_cast<int>(floor(input_[it].x / _Delta) -
                                static_cast<double>(min_b_[0]));
    int ijk1 = static_cast<int>(floor(input_[it].y / _Delta) -
                                static_cast<double>(min_b_[1]));
    int ijk2 = static_cast<int>(floor(input_[it].z / _Delta) -
                                static_cast<double>(min_b_[2]));

    // Compute the centroid leaf index
    int idx = ijk0 * divb_mul_[0] + ijk1 * divb_mul_[1] + ijk2 * divb_mul_[2];
    index_vector.push_back(cloud_point_index_idx(static_cast<size_t>(idx), it));
  }
  auto compare = [](const cloud_point_index_idx &lhs,
                    const cloud_point_index_idx &rhs) {
    return (lhs.idx == rhs.idx);
  };

  auto predict = [](const cloud_point_index_idx &lhs,
                    const cloud_point_index_idx &rhs) {
    return lhs.idx < rhs.idx;
  };

  std::sort(index_vector.begin(), index_vector.end(),
            std::less<cloud_point_index_idx>());

  // Third pass: count output cells
  // we need to skip all the same, adjacent idx values
  size_t total = 0;
  size_t index = 0;
  // first_and_last_indices_vector[i] represents the index in index_vector of
  // the first point in index_vector belonging to the voxel which corresponds to
  // the i-th output point, and of the first point not belonging to.
  std::vector<std::pair<size_t, size_t>> first_and_last_indices_vector;
  // Worst case size
  first_and_last_indices_vector.reserve(index_vector.size());
  while (index < index_vector.size()) {
    size_t i = index + 1;
    while (i < index_vector.size() &&
           index_vector[i].idx == index_vector[index].idx)
      ++i;
    if (i - index >= 0) {
      ++total;
      first_and_last_indices_vector.push_back(
          std::pair<size_t, size_t>(index, i));
    }
    index = i;
  }
  output.resize(total);

  index = 0;
  for (size_t cp = 0; cp < first_and_last_indices_vector.size(); ++cp) {
    // calculate centroid - sum values from all input points, that have the same
    // idx value in index_vector array
    size_t first_index = first_and_last_indices_vector[cp].first;
    size_t last_index = first_and_last_indices_vector[cp].second;

    double x(0.0f), y(0.0f), z(0.0f);

    // fill in the accumulator with leaf points
    size_t counter(0);
    for (size_t li = first_index; li < last_index; ++li) {
      x = x + input_[index_vector[li].cloud_point_index].x;
      y = y + input_[index_vector[li].cloud_point_index].y;
      z = z + input_[index_vector[li].cloud_point_index].z;
      ++counter;
    }

    output[index].x = x / counter;
    output[index].y = y / counter;
    output[index].z = z / counter;

    ++index;
  }
  output.shrink_to_fit();
}

double Filter3D::getMean(size_t _k, pointCloud input_) {
  KDtreeFlann tree;

  tree.setInputCloud(input_);
  std::vector<size_t> indices;
  size_t size = input_.size();
  double std_mul_ = 1.0;

  // The arrays to be used
  std::vector<size_t> nn_indices(_k);
  std::vector<double> nn_dists(_k);
  std::vector<double> distances(size);
  indices.resize(size);
  size_t oii = 0,
         rii =
             0; // oii = output indices iterator, rii = removed indices iterator

  // First pass: Compute the mean distances for all points with respect to their
  // k nearest neighbours
  size_t valid_distances = 0;
  for (size_t iii = 0; iii < size; ++iii) // iii = input indices iterator
  {
    // Perform the nearest k search
    tree.knnSearch(input_[iii], _k, nn_indices, nn_dists);

    // Calculate the mean distance to its neighbours
    double dist_sum = 0.0;
    for (size_t it = 1; it < _k; ++it) // k = 0 is the query point
      dist_sum += sqrt(nn_dists[it]);
    distances[iii] = static_cast<double>(dist_sum / _k);
    valid_distances++;
  }

  // Estimate the mean and the standard deviation of the distance vector
  double sum = 0, sq_sum = 0;
  sum = std::accumulate(distances.begin(), distances.end(), 0);
  sq_sum = std::accumulate(
      distances.begin(), distances.end(), 0,
      [](double &lhs, double &rhs) { return (lhs + (rhs * rhs)); });

  double mean = sum / static_cast<double>(valid_distances);
  double variance =
      (sq_sum - sum * sum / static_cast<double>(valid_distances)) /
      (static_cast<double>(valid_distances) - 1);
  double stddev = sqrt(variance);
  return (mean + std_mul_ * stddev);
}

void Filter3D::MakeSmoothPointCloud(size_t _k, double smoothingFactor,
                                    pointCloud input_, pointCloud &output) {
  KDtreeFlann tree;

  tree.setInputCloud(input_);
  std::vector<size_t> indices;
  size_t size = input_.size();
  double std_mul_ = 1.0;

  // The arrays to be used
  std::vector<double> avgX(size), avgY(size), avgZ(size);
  std::vector<size_t> nn_indices(_k);
  std::vector<double> nn_dists(_k);
  std::vector<size_t> indices_radius(1);
  std::vector<double> dists_radius(1);
  std::vector<double> distances(size);
  indices.resize(size);
  size_t oii = 0,
         rii =
             0; // oii = output indices iterator, rii = removed indices iterator

  // First pass: Compute the mean distances for all points with respect to their
  // k nearest neighbours
  size_t valid_distances = 0;
  for (size_t iii = 0; iii < size; ++iii) // iii = input indices iterator
  {
    // Perform the nearest k search
    tree.knnSearch(input_[iii], _k, nn_indices, nn_dists);

    // Calculate the mean distance to its neighbours
    double dist_sum = 0.0;
    for (int it = 1; it < _k; ++it) // k = 0 is the query point
      dist_sum += sqrt(nn_dists[it]);
    distances[iii] = static_cast<double>(dist_sum / _k);
    valid_distances++;
  }

  // Estimate the mean and the standard deviation of the distance vector
  double sum = 0, sq_sum = 0;

  sum = std::accumulate(distances.begin(), distances.end(), 0);
  sq_sum = std::accumulate(
      distances.begin(), distances.end(), 0,
      [](double &lhs, double &rhs) { return (lhs + (rhs * rhs)); });

  double mean = sum / static_cast<double>(valid_distances);

  double variance =
      (sq_sum - sum * sum / static_cast<double>(valid_distances)) /
      (static_cast<double>(valid_distances) - 1);

  double stddev = sqrt(variance);

  double radius = 0.0f;

  if (smoothingFactor < 1)
    radius = (mean + std_mul_ * stddev);
  else
    radius = smoothingFactor * (mean + std_mul_ * stddev);

  size_t counter = 0;
  for (size_t i = 0; i < size; i++) {
    if (tree.radiusSearch(input_[i], radius, indices_radius, dists_radius) >
        0) {
      size_t numberOfIndices = indices_radius.size();
      if (numberOfIndices > 10) {
        for (size_t j = 0; j < numberOfIndices; j++) {
          // Accumulating the co-ordinates of the points which comes in the
          // volume sphere
          avgX[counter] = avgX[counter] + input_[indices_radius[j]].x;
          avgY[counter] = avgY[counter] + input_[indices_radius[j]].y;
          avgZ[counter] = avgZ[counter] + input_[indices_radius[j]].z;
        }
        // New point cloud with average points
        avgX[counter] = avgX[counter] / numberOfIndices;
        avgY[counter] = avgY[counter] / numberOfIndices;
        avgZ[counter] = avgZ[counter] / numberOfIndices;
        counter++;
      }
    }
  }

  // Resize again since that the removed points will change the size.
  avgX.resize(counter);
  avgY.resize(counter);
  avgZ.resize(counter);

  // Saving the reduced pointCloud in to new PCD file.
  output.resize(counter);

  for (int i = 0; i < counter; i++) {
    output[i].x = avgX[i];
    output[i].y = avgY[i];
    output[i].z = avgZ[i];
  }
}

void Filter3D::euclideanClustering(double radius, size_t minClusterSize,
                                   size_t maxClusterSize, pointCloud input_,
                                   std::vector<pointCloud> &output) {
  KDtreeFlann tree;

  tree.setInputCloud(input_);
  std::vector<bool> alreadyProcesed(input_.size(), false);
  std::vector<size_t> raidusIndices;
  std::vector<double> raidusDistances;
  std::vector<std::vector<int>> clusters;
  size_t counter = 0;

  for (size_t i = 0; i < input_.size(); i++) {
    if (alreadyProcesed[i])
      continue;

    std::vector<size_t> clusterPoints;
    size_t index = 0;
    clusterPoints.push_back(i);

    alreadyProcesed[i] = true;

    while (index < static_cast<int>(clusterPoints.size())) {
      if (!tree.radiusSearch(input_[clusterPoints[index]], radius,
                             raidusIndices, raidusDistances)) {
        index++;
        continue;
      }

      for (size_t j = 0; j < raidusIndices.size(); j++) {
        if (alreadyProcesed[raidusIndices[j]])
          continue;

        clusterPoints.push_back(raidusIndices[j]);
        alreadyProcesed[raidusIndices[j]] = true;
      }
      index++;
    }

    if (clusterPoints.size() >= minClusterSize &&
        clusterPoints.size() <= maxClusterSize) {
      std::vector<int> sortedCloud;
      pointCloud intermediattPointCloud;
      sortedCloud.resize(clusterPoints.size());
      for (size_t j = 0; j < clusterPoints.size(); j++)
        sortedCloud[j] = clusterPoints[j];

      std::sort(sortedCloud.begin(), sortedCloud.end());
      sortedCloud.erase(std::unique(sortedCloud.begin(), sortedCloud.end()),
                        sortedCloud.end());
      intermediattPointCloud.resize(sortedCloud.size());
      for (size_t k = 0; k < sortedCloud.size(); k++) {
        intermediattPointCloud[k].x = input_[sortedCloud[k]].x;
        intermediattPointCloud[k].y = input_[sortedCloud[k]].y;
        intermediattPointCloud[k].z = input_[sortedCloud[k]].z;
      }
      output.push_back(intermediattPointCloud);
    }
  }
  sort(output.begin(), output.end(), [](pointCloud &lhs, pointCloud &rhs) {
    return lhs.size() > rhs.size();
  });
  output.shrink_to_fit();
}

void Filter3D::euclideanClustering(double radius, size_t minClusterSize,
                                   size_t maxClusterSize, pointCloud input_,
                                   std::vector<std::vector<int>> &output) {
  KDtreeFlann tree;

  tree.setInputCloud(input_);
  std::vector<bool> alreadyProcesed(input_.size(), false);
  std::vector<size_t> raidusIndices;
  std::vector<double> raidusDistances;
  std::vector<std::vector<int>> clusters;
  size_t counter = 0;

  for (size_t i = 0; i < input_.size(); i++) {
    if (alreadyProcesed[i])
      continue;

    std::vector<size_t> clusterPoints;
    size_t index = 0;
    clusterPoints.push_back(i);

    alreadyProcesed[i] = true;

    while (index < static_cast<int>(clusterPoints.size())) {
      if (!tree.radiusSearch(input_[clusterPoints[index]], radius,
                             raidusIndices, raidusDistances)) {
        index++;
        continue;
      }

      for (size_t j = 0; j < raidusIndices.size(); j++) {
        if (alreadyProcesed[raidusIndices[j]])
          continue;

        clusterPoints.push_back(raidusIndices[j]);
        alreadyProcesed[raidusIndices[j]] = true;
      }
      index++;
    }

    if (clusterPoints.size() >= minClusterSize &&
        clusterPoints.size() <= maxClusterSize) {
      std::vector<int> sortedCloud;
      sortedCloud.resize(clusterPoints.size());
      for (int j = 0; j < clusterPoints.size(); j++)
        sortedCloud[j] = clusterPoints[j];

      std::sort(sortedCloud.begin(), sortedCloud.end());
      sortedCloud.erase(std::unique(sortedCloud.begin(), sortedCloud.end()),
                        sortedCloud.end());
      output.push_back(sortedCloud);
    }
  }
  sort(output.begin(), output.end(),
       [](std::vector<int> &lhs, std::vector<int> &rhs) {
         return lhs.size() > rhs.size();
       });
  output.shrink_to_fit();
}

void Filter3D::findCentroidAndCovarianceMatrix(
    pointCloud cloud, Eigen::Matrix3f &covariance_matrix,
    Eigen::Matrix<double, 4, 1> &centroid) {
  // initialize matrix in row form to save the computations
  Eigen::Matrix<double, 1, 9, Eigen::RowMajor> accu =
      Eigen::Matrix<double, 1, 9, Eigen::RowMajor>::Zero();
  size_t point_count;
  point_count = cloud.size();
  // For each point in the cloud
  for (size_t i = 0; i < point_count; ++i) {
    accu[0] += cloud[i].x * cloud[i].x;
    accu[1] += cloud[i].x * cloud[i].y;
    accu[2] += cloud[i].x * cloud[i].z;
    accu[3] += cloud[i].y * cloud[i].y; // 4
    accu[4] += cloud[i].y * cloud[i].z; // 5
    accu[5] += cloud[i].z * cloud[i].z; // 8
    accu[6] += cloud[i].x;
    accu[7] += cloud[i].y;
    accu[8] += cloud[i].z;
  }

  accu /= static_cast<double>(point_count);
  if (point_count != 0) {
    centroid[0] = accu[6];
    centroid[1] = accu[7];
    centroid[2] = accu[8];
    centroid[3] = 1;
    covariance_matrix.coeffRef(0) = accu[0] - accu[6] * accu[6];
    covariance_matrix.coeffRef(1) = accu[1] - accu[6] * accu[7];
    covariance_matrix.coeffRef(2) = accu[2] - accu[6] * accu[8];
    covariance_matrix.coeffRef(4) = accu[3] - accu[7] * accu[7];
    covariance_matrix.coeffRef(5) = accu[4] - accu[7] * accu[8];
    covariance_matrix.coeffRef(8) = accu[5] - accu[8] * accu[8];
    covariance_matrix.coeffRef(3) = covariance_matrix.coeff(1);
    covariance_matrix.coeffRef(6) = covariance_matrix.coeff(2);
    covariance_matrix.coeffRef(7) = covariance_matrix.coeff(5);
  }
}

void Filter3D::findNormals(pointCloud cloud, size_t numberOfneighbour,
                           std::vector<normalsAndCurvature> &output) {
  Eigen::Matrix3f covariance_matrix;
  Eigen::Matrix<double, 4, 1> centroid;
  pointCloud processingCloud(numberOfneighbour);
  KDtreeFlann tree(cloud);
  std::vector<size_t> pointIdxKSearch(numberOfneighbour);
  std::vector<double> pointKSquaredDistance(numberOfneighbour);
  output.resize(cloud.size());
  for (size_t i = 0; i < cloud.size(); i++) {
    tree.knnSearch(cloud[i], numberOfneighbour, pointIdxKSearch,
                   pointKSquaredDistance);
    for (size_t j = 0; j < numberOfneighbour; j++) {
      processingCloud[j].x = cloud[pointIdxKSearch[j]].x;
      processingCloud[j].y = cloud[pointIdxKSearch[j]].y;
      processingCloud[j].z = cloud[pointIdxKSearch[j]].z;
    }

    findCentroidAndCovarianceMatrix(processingCloud, covariance_matrix,
                                    centroid);

    Eigen::Matrix3f::Scalar scale = covariance_matrix.cwiseAbs().maxCoeff();
    Eigen::Matrix3f sacledCVM = covariance_matrix / scale;

    // Extract the smallest eigenvalue and its eigenvector
    Eigen::EigenSolver<Eigen::Matrix3f> es(sacledCVM);

    std::vector<double> eval(3);
    eval[0] = es.eigenvalues()[0].real();
    eval[1] = es.eigenvalues()[1].real();
    eval[2] = es.eigenvalues()[2].real();

    std::vector<double>::iterator result =
        std::min_element(std::begin(eval), std::end(eval));
    int minEvalIndex = std::distance(std::begin(eval), result);

    Eigen::VectorXcf v = es.eigenvectors().col(minEvalIndex);
    output[i].nx = v[0].real();
    output[i].ny = v[1].real();
    output[i].nz = v[2].real();
    if (((output[i].nx * cloud[i].x) + (output[i].ny * cloud[i].y) +
         (output[i].nz * cloud[i].z)) > 0) {
      output[i].nx *= -1;
      output[i].ny *= -1;
      output[i].nz *= -1;
    }
    // Compute the curvature surface change
    double eig_sum =
        sacledCVM.coeff(0) + sacledCVM.coeff(4) + sacledCVM.coeff(8);
    if (eig_sum != 0)
      output[i].curvature =
          fabsf(es.eigenvalues()[minEvalIndex].real() / eig_sum);
    else
      output[i].curvature = 0;
  }
}

void Filter3D::SetPasstroughLimits_X(double x_min, double x_max) {
  if (x_min <= x_max) {
    x_limits.first = x_min;
    x_limits.second = x_max;
    x_passthough_limits_set = true;
  }
}

void Filter3D::SetPasstroughLimits_Y(double y_min, double y_max) {
  if (y_min <= y_max) {
    y_limits.first = y_min;
    y_limits.second = y_max;
    y_passthough_limits_set = true;
  }
}

void Filter3D::SetPasstroughLimits_Z(double z_min, double z_max) {
  if (z_min <= z_max) {
    z_limits.first = z_min;
    z_limits.second = z_max;
    z_passthough_limits_set = true;
  }
}

void Filter3D::PassthroughFilter(pointCloud &input, pointCloud &output) {
  pointCloud points = input;
  output.resize(points.size());
  pointCloud::iterator it = output.begin();

  if (x_passthough_limits_set && y_passthough_limits_set &&
      z_passthough_limits_set) {
    it = std::copy_if(points.begin(), points.end(), output.begin(),
                      [this](const Point &p) {
                        return (InRange(p.x, x_limits.first, x_limits.second) &&
                                InRange(p.y, y_limits.first, y_limits.second) &&
                                InRange(p.z, z_limits.first, z_limits.second));
                      });
  } else if (x_passthough_limits_set && y_passthough_limits_set) {
    it = std::copy_if(points.begin(), points.end(), output.begin(),
                      [this](const Point &p) {
                        return (InRange(p.x, x_limits.first, x_limits.second) &&
                                InRange(p.y, y_limits.first, y_limits.second));
                      });
  } else if (x_passthough_limits_set && z_passthough_limits_set) {
    it = std::copy_if(points.begin(), points.end(), output.begin(),
                      [this](const Point &p) {
                        return (InRange(p.x, x_limits.first, x_limits.second) &&
                                InRange(p.z, z_limits.first, z_limits.second));
                      });
  } else if (y_passthough_limits_set && z_passthough_limits_set) {
    it = std::copy_if(points.begin(), points.end(), output.begin(),
                      [this](const Point &p) {
                        return (InRange(p.y, y_limits.first, y_limits.second) &&
                                InRange(p.z, z_limits.first, z_limits.second));
                      });
  } else if (x_passthough_limits_set) {
    it = std::copy_if(points.begin(), points.end(), output.begin(),
                      [this](const Point &p) {
                        return (InRange(p.x, x_limits.first, x_limits.second));
                      });
  } else if (y_passthough_limits_set) {
    it = std::copy_if(points.begin(), points.end(), output.begin(),
                      [this](const Point &p) {
                        return (InRange(p.y, y_limits.first, y_limits.second));
                      });
  } else if (z_passthough_limits_set) {
    it = std::copy_if(points.begin(), points.end(), output.begin(),
                      [this](const Point &p) {
                        return (InRange(p.z, z_limits.first, z_limits.second));
                      });
  }

  output.resize(std::distance(output.begin(), it));

  x_passthough_limits_set = false;
  y_passthough_limits_set = false;
  z_passthough_limits_set = false;
}