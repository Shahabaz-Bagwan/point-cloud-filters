
#pragma once

#include "PointDefinition.hpp"
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

struct Correspondence
{
  /** \brief Index of the query (source) point. */
  unsigned long indexQuery;
  /** \brief Index of the matching (target) point. Set to -1 if no
   * correspondence found. */
  unsigned long indexMatch;
  /** \brief Distance between the corresponding points, or the weight denoting
   * the confidence in correspondence estimation */
  double distanceOrWeight;

  /** \brief Standard constructor.
   * Sets \ref index_query to 0, \ref index_match to -1, and \ref distance to
   * FLT_MAX.
   */
  inline Correspondence()
    : indexQuery( 0 ), indexMatch( -1 ),
      distanceOrWeight( std::numeric_limits< double >::max() )
  {
  }

  Correspondence( const Correspondence& )            = default;
  Correspondence( Correspondence&& )                 = delete;
  Correspondence& operator=( const Correspondence& ) = default;
  Correspondence& operator=( Correspondence&& )      = delete;
  inline Correspondence( int indexQuery, int indexMatch,
                         double distanceOrWeight )
    : indexQuery( indexQuery ), indexMatch( indexMatch ),
      distanceOrWeight( distanceOrWeight )
  {
  }

  virtual ~Correspondence() = default;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

namespace PCF {
  class IterativeClosestPoint
  {
  public:
    IterativeClosestPoint();

    ~IterativeClosestPoint() = default;

    [[nodiscard]] bool isConverged() const { return converged_; }

    void align( double maxDistance, size_t maxIteration, pointCloud& source,
                pointCloud target, pointCloud& output );

    Eigen::Matrix4d getTxMatrix() { return transformation_matrix_; }

  private:
    Eigen::Matrix4d transformation_matrix_ = Eigen::Matrix4d::Identity();
    double correspondences_prev_mse_;

    inline bool hasConverged( size_t iterations, size_t maxIterations,
                              std::vector< Correspondence > correspondences );

    inline void
      determineCorrespondences( std::vector< Correspondence >& correspondences,
                                double maxDistance, pointCloud input,
                                pointCloud target );

    inline void estimateRigidTransformation(
      pointCloud input, pointCloud target,
      std::vector< Correspondence > correspondences,
      Eigen::Matrix4d& transformationMatrix );

    inline void transformCloud( pointCloud& input, pointCloud& output,
                                const Eigen::Matrix4d& transform );

    bool converged_ = false;
  };
} // namespace PCF
