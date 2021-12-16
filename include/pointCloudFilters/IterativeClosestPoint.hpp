
#pragma once

#include "KdtreeFlann.hpp"
#include "PointDefinition.hpp"
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

namespace {
  struct Correspondence
  {
    /** \brief Index of the query (source) point. */
    int index_query;
    /** \brief Index of the matching (target) point. Set to -1 if no
     * correspondence found. */
    int index_match;
    /** \brief Distance between the corresponding points, or the weight denoting
     * the confidence in correspondence estimation */
    union
    {
      double distance;
      double weight;
    };

    /** \brief Standard constructor.
     * Sets \ref index_query to 0, \ref index_match to -1, and \ref distance to
     * FLT_MAX.
     */
    inline Correspondence()
      : index_query( 0 ), index_match( -1 ),
        distance( std::numeric_limits< double >::max() )
    {}

    inline Correspondence( int _index_query, int _index_match,
                           double _distance )
      : index_query( _index_query ), index_match( _index_match ),
        distance( _distance )
    {}

    virtual ~Correspondence() {}

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };
} // namespace
namespace PCF {
  class IterativeClosestPoint
  {
  public:
    IterativeClosestPoint();

    ~IterativeClosestPoint();

    bool converged = false;

    void align( double max_distance, size_t maxIteration, pointCloud& source,
                pointCloud target, pointCloud& output );

  private:
    double correspondences_prev_mse_;

    inline bool hasConverged( size_t iterations_, size_t max_iterations_,
                              Eigen::Matrix4d& transformation_,
                              std::vector< Correspondence > correspondences );

    inline void
      determineCorrespondences( std::vector< Correspondence >& correspondences,
                                double max_distance, pointCloud input,
                                pointCloud target );

    inline void estimateRigidTransformation(
      pointCloud input, pointCloud target,
      std::vector< Correspondence > correspondences,
      Eigen::Matrix4d& transformation_matrix );

    inline void transformCloud( pointCloud& input, pointCloud& output,
                                const Eigen::Matrix4d& transform );
  };
} // namespace PCF