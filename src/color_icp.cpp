// Color ICP Registration
// Hanzhe Teng, Feb 2022

#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/search/kdtree.h>
#include <pcl/search/impl/kdtree.hpp> // needed for L2 distance; skip PCL_NO_PRECOMPILE
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>  // needed for L1 distance

#include <Eigen/Geometry>

#include "color_icp/helper.h"
#include "color_icp/remove_nan.h"
#include "color_icp/yaml.h"


namespace Eigen {
    typedef Eigen::Matrix<float, 6, 6> Matrix6f;
    typedef Eigen::Matrix<float, 6, 1> Vector6f;
    typedef Eigen::Matrix<float, 6, -1> Matrix6Xf;
} // namespace Eigen


class ColorICP {
  public:
    using PointT = pcl::PointXYZRGBNormal;
    using NormalT = pcl::PointXYZRGBNormal;
    using PointCloudPtr = pcl::PointCloud<PointT>::Ptr;

    ColorICP ();
    ~ColorICP () {}

  protected:
    void removeNaNPoints (const PointCloudPtr& cloud);
    void downSampleVoxelGrids (const PointCloudPtr& cloud);
    void estimateNormals (const PointCloudPtr& cloud);
    void copyPointCloud (const PointCloudPtr& cloud_in, const std::vector<int>& indices, PointCloudPtr& cloud_out);
    void visualizeRegistration (const PointCloudPtr& source, const PointCloudPtr& source_transformed, const PointCloudPtr& target);
    
    Eigen::Matrix4f PCLICP (const PointCloudPtr& source, const PointCloudPtr& target);
    Eigen::Matrix4f ClassicICPRegistration (const PointCloudPtr& source, const PointCloudPtr& target);
    Eigen::Matrix4f ColorICPRegistration (const PointCloudPtr& source, const PointCloudPtr& target);
    Eigen::Matrix4f GaussNewton (const PointCloudPtr& source, const PointCloudPtr& target);
    Eigen::Matrix4f TransformVector6fToMatrix4f (const Eigen::Vector6f& input);

  private:
    pcl::search::KdTree<PointT, pcl::KdTreeFLANN<PointT, flann::L2_Simple<float>>>::Ptr kdtree_;
    pcl::PointCloud<PointT>::Ptr source_cloud_;
    pcl::PointCloud<PointT>::Ptr target_cloud_;
    Eigen::Matrix4f transformation_;
    Yaml::Node params_;
};


ColorICP::ColorICP ()
    : source_cloud_ (new pcl::PointCloud<PointT>)
    , target_cloud_ (new pcl::PointCloud<PointT>)
    , transformation_ (Eigen::Matrix4f::Identity ())
    , kdtree_ (new pcl::search::KdTree<PointT, pcl::KdTreeFLANN<PointT, flann::L2_Simple<float>>>) {

  // load point clouds
  Yaml::Parse (params_, "../config/params.yaml");
  std::string src_filename = params_["source_cloud"].As<std::string> ();
  std::string tar_filename = params_["target_cloud"].As<std::string> ();
  if (my::loadPointCloud<PointT> (src_filename, *source_cloud_))
    std::cout << "[source] Loaded " << source_cloud_->size() << " points\n";
  if (my::loadPointCloud<PointT> (tar_filename, *target_cloud_))
    std::cout << "[target] Loaded " << target_cloud_->size() << " points\n";

  // pre-processing
  removeNaNPoints (source_cloud_);
  removeNaNPoints (target_cloud_);
  downSampleVoxelGrids (source_cloud_);
  downSampleVoxelGrids (target_cloud_);
  estimateNormals (source_cloud_);
  estimateNormals (target_cloud_);

  // registration
  std::string icp_method = params_["icp_method"].As<std::string> ();
  if (icp_method == "pcl")
    transformation_ = PCLICP (source_cloud_, target_cloud_);
  else if (icp_method == "classic")
    transformation_ = ClassicICPRegistration(source_cloud_, target_cloud_);
  else if (icp_method == "color")
    transformation_ = ColorICPRegistration(source_cloud_, target_cloud_);
  else
    PCL_ERROR("no registration method is selected");
  std::cout << "Estimated transformation " << std::endl << transformation_ << std::endl;

  // visualization
  pcl::PointCloud<PointT>::Ptr source_cloud_transformed (new pcl::PointCloud<PointT>);
  pcl::transformPointCloud (*source_cloud_, *source_cloud_transformed, transformation_);
  visualizeRegistration(source_cloud_, source_cloud_transformed, target_cloud_);
}


void ColorICP::removeNaNPoints (const PointCloudPtr& cloud) {
  std::vector<int> nan_idx;
  pcl::removeNaNFromPointCloudBruteForce (*cloud, *cloud, nan_idx);
  std::cout << "Contained " << cloud->size () << " points after removing NaN points\n";
  pcl::removeNaNRGBFromPointCloud (*cloud, *cloud, nan_idx);
  std::cout << "Contained " << cloud->size () << " points after removing NaN RGB points\n";
}


void ColorICP::downSampleVoxelGrids (const PointCloudPtr& cloud) {
  float resolution = params_["voxel_resolution"].As<float> ();
  pcl::VoxelGrid<PointT> sor;
  sor.setLeafSize (resolution, resolution, resolution);
  sor.setInputCloud (cloud);
  sor.filter (*cloud);
  std::cout << "Downsampled to " << cloud->size () << " points\n";
}


void ColorICP::estimateNormals (const PointCloudPtr& cloud) {
  pcl::NormalEstimation<PointT, NormalT> ne;
  ne.setSearchMethod (kdtree_);
  ne.setRadiusSearch (params_["normal_est_radius"].As<float> ());
  ne.setInputCloud (cloud);
  ne.compute (*cloud);
  std::cout << "Computed " << cloud->size () << " Normals\n";
  std::vector<int> nan_idx;
  pcl::removeNaNNormalsFromPointCloud (*cloud, *cloud, nan_idx);
  std::cout << "Contained " << cloud->size () << " points after removing NaN normals\n";
}


void ColorICP::copyPointCloud (const PointCloudPtr& cloud_in, const std::vector<int>& indices, PointCloudPtr& cloud_out) {
  // allocate enough space and copy the basics
  cloud_out->points.resize (indices.size ());
  cloud_out->header   = cloud_in->header;
  cloud_out->width    = static_cast<uint32_t>(indices.size ());
  cloud_out->height   = 1;
  cloud_out->is_dense = cloud_in->is_dense;
  cloud_out->sensor_orientation_ = cloud_in->sensor_orientation_;
  cloud_out->sensor_origin_ = cloud_in->sensor_origin_;

  // iterate over each point
  for (size_t i = 0; i < indices.size (); ++i)
    cloud_out->points[i] = cloud_in->points[indices[i]];
}


void ColorICP::visualizeRegistration (const PointCloudPtr& source,
                                      const PointCloudPtr& source_transformed, 
                                      const PointCloudPtr& target) {
  // add point clouds to the viewer
  pcl::visualization::PCLVisualizer visualizer;
  pcl::visualization::PointCloudColorHandlerCustom<PointT> source_color_handler (source, 255, 255, 0);
  pcl::visualization::PointCloudColorHandlerCustom<PointT> source_transformed_color_handler (source_transformed, 255, 255, 255);
  pcl::visualization::PointCloudColorHandlerCustom<PointT> target_color_handler (target, 0, 255, 255);
  visualizer.addPointCloud (source, source_color_handler, "source cloud");
  visualizer.addPointCloud (source_transformed, source_transformed_color_handler, "source cloud transformed");
  visualizer.addPointCloud (target, target_color_handler, "target cloud");
  
  while (!visualizer.wasStopped ()) {
    visualizer.spinOnce ();
    pcl_sleep(0.01);
  }
}


Eigen::Matrix4f ColorICP::PCLICP (const PointCloudPtr& source, const PointCloudPtr& target) {
  pcl::PointCloud<PointT> registration_output;
  pcl::IterativeClosestPoint<PointT, PointT> icp;
  icp.setMaxCorrespondenceDistance (params_["icp_max_corres_dist"].As<float> ());
  icp.setTransformationEpsilon (params_["icp_transformation_epsilon"].As<float> ());
  icp.setMaximumIterations (params_["icp_max_iterations"].As<int> ());
  icp.setInputSource (source_cloud_);
  icp.setInputTarget (target_cloud_);
  icp.align (registration_output);
  return icp.getFinalTransformation ();
}


// refer to pcl/registration/impl/icp.hpp and transformation_estimation_svd.hpp
// exactly the same behavior and performance as using PCLICP()
Eigen::Matrix4f ColorICP::ClassicICPRegistration (const PointCloudPtr& source, const PointCloudPtr& target) {
  // initialization
  int iteration = params_["icp_max_iterations"].As<int> ();
  float distance_threshold = params_["icp_max_corres_dist"].As<float> (); // icp.setMaxCorrespondenceDistance
  int cloud_size = static_cast<int> (source->size());
  Eigen::Matrix4f final_transformation = Eigen::Matrix4f::Identity();
  pcl::PointCloud<PointT>::Ptr source_trans (new pcl::PointCloud<PointT>);

  // build K-d tree for target cloud
  pcl::search::KdTree<PointT>::Ptr kdtree (new pcl::search::KdTree<PointT>);
  kdtree->setInputCloud(target);
  std::vector<int> indices (1);    // for nearestKSearch
  std::vector<float> sq_dist (1);  // for nearestKSearch

  // repeat until convergence
  for (int t = 0; t < iteration; ++t) {
    // transform source using estimated transformation
    pcl::transformPointCloud<PointT> (*source, *source_trans, final_transformation);

    // visualize source_trans in each step if needed

    // find correspondences in target
    std::vector<std::pair<int, int>> correspondences;
    for (int i = 0; i < cloud_size; ++i) {
      kdtree->nearestKSearch(source_trans->points[i], 1, indices, sq_dist);
      if (sq_dist[0] > distance_threshold * distance_threshold) // skip if too far
        continue;
      correspondences.push_back({i, indices[0]});
    }

    // convert to Eigen format
    int idx = 0;
    Eigen::Matrix<float, 3, Eigen::Dynamic> cloud_src (3, correspondences.size());
    Eigen::Matrix<float, 3, Eigen::Dynamic> cloud_tgt (3, correspondences.size());
    for (const auto& corres : correspondences) {
      cloud_src (0, idx) = source_trans->points[corres.first].x;
      cloud_src (1, idx) = source_trans->points[corres.first].y;
      cloud_src (2, idx) = source_trans->points[corres.first].z;
      cloud_tgt (0, idx) = target->points[corres.second].x;
      cloud_tgt (1, idx) = target->points[corres.second].y;
      cloud_tgt (2, idx) = target->points[corres.second].z;
      ++idx;
    }

    // skip a few steps here for simplicity, such as
    // check convergence (if trans update < required epsilon)
    // check if cloud_src and cloud_tgt are valid (>0 or >3?)

    // solve using Umeyama's algorithm (SVD)
    Eigen::Matrix4f transformation = Eigen::umeyama (cloud_src, cloud_tgt, false);
    final_transformation = transformation * final_transformation;
    std::cout << "it = " << t << "; cloud size = " << cloud_size << "; idx = " << idx << std::endl;
    std::cout << "current transformation estimation" << std::endl << final_transformation << std::endl;
  }

  return final_transformation;
}


// Implementation according to Section 4.3 in the paper: Colored Point Cloud Registration Revisited, ICCV 2017
// https://github.com/isl-org/Open3D/blob/master/cpp/open3d/pipelines/registration/ColoredICP.cpp
// https://github.com/isl-org/Open3D/blob/master/cpp/open3d/utility/Eigen.cpp
Eigen::Matrix4f ColorICP::ColorICPRegistration (const PointCloudPtr& source, const PointCloudPtr& target) {
  // initialization
  int iteration = params_["icp_max_iterations"].As<float> ();
  float distance_threshold = params_["icp_max_corres_dist"].As<float> (); // icp.setMaxCorrespondenceDistance
  int cloud_size = static_cast<int> (source->size());
  Eigen::Matrix4f final_transformation = Eigen::Matrix4f::Identity();
  pcl::PointCloud<PointT>::Ptr source_trans (new pcl::PointCloud<PointT>);

  // build K-d tree for target cloud
  pcl::search::KdTree<PointT>::Ptr kdtree (new pcl::search::KdTree<PointT>);
  kdtree->setInputCloud(target);
  std::vector<int> indices (1);    // for nearestKSearch
  std::vector<float> sq_dist (1);  // for nearestKSearch

  // repeat until convergence
  for (int t = 0; t < iteration; ++t) {
    // transform source using estimated transformation
    pcl::transformPointCloud<PointT> (*source, *source_trans, final_transformation);
    
    // find correspondences
    std::vector<int> indices_src;
    std::vector<int> indices_tgt;
    for (int i = 0; i < cloud_size; ++i) {
      kdtree->nearestKSearch(source_trans->points[i], 1, indices, sq_dist);
      if (sq_dist[0] > distance_threshold * distance_threshold) // skip if too far
        continue;
      indices_src.push_back(i);
      indices_tgt.push_back(indices[0]);
    }

    // copy point clouds
    pcl::PointCloud<PointT>::Ptr cloud_src (new pcl::PointCloud<PointT>);
    pcl::PointCloud<PointT>::Ptr cloud_tgt (new pcl::PointCloud<PointT>);
    copyPointCloud (source_trans, indices_src, cloud_src);
    copyPointCloud (target, indices_tgt, cloud_tgt);
    std::cout << "it = " << t << "; cloud size = " << cloud_size << "; selected size = " << indices_src.size() << std::endl;

    // solve using Gauss-Newton method
    Eigen::Matrix4f transformation = GaussNewton (cloud_src, cloud_tgt);
    final_transformation = transformation * final_transformation;
    // std::cout << "transformation in this iteration" << std::endl << transformation << std::endl;
    std::cout << "current transformation estimation" << std::endl << final_transformation << std::endl;
  }

  return final_transformation;
}


// Note: current implementation considers geometric component only (not yet color component)
Eigen::Matrix4f ColorICP::GaussNewton (const PointCloudPtr& source, const PointCloudPtr& target) {
  int size = static_cast<int> (source->size());
  Eigen::Matrix6Xf Jacobian (6, size); // alpha, beta, gamma, a, b, c as in the linearized transformation matrix
  Eigen::VectorXf Residual (size);     // see Eq. 20 in the paper

  for (int i = 0; i < size; ++i) {
    const Eigen::Vector3f& q = source->points[i].getVector3fMap();
    const Eigen::Vector3f& p = target->points[i].getVector3fMap();
    const Eigen::Vector3f& np = target->points[i].getNormalVector3fMap();
    Residual (i) = (q - p).dot(np);
    Jacobian.block<3, 1>(0, i) = q.cross(np);
    Jacobian.block<3, 1>(3, i) = np;
  }

  Eigen::Matrix6f JTJ = Jacobian * Jacobian.transpose();  // Jacobian herein has already been transposed
  Eigen::Vector6f JTr = Jacobian * Residual;
  Eigen::Vector6f X = JTJ.ldlt().solve(-JTr);  // solve a system of linear equations in the form: AX = b
  return TransformVector6fToMatrix4f(X);
}


Eigen::Matrix4f ColorICP::TransformVector6fToMatrix4f(const Eigen::Vector6f& input) {
  Eigen::Matrix4f output;
  output.setIdentity();
  // AngleAxis representation implicitly maps the linearized matrix to SO(3)
  output.block<3, 3>(0, 0) = (Eigen::AngleAxisf(input(2), Eigen::Vector3f::UnitZ()) *
                              Eigen::AngleAxisf(input(1), Eigen::Vector3f::UnitY()) *
                              Eigen::AngleAxisf(input(0), Eigen::Vector3f::UnitX())).matrix();
  output.block<3, 1>(0, 3) = input.block<3, 1>(3, 0);
  return output;
}


int main(int argc, char** argv) {
  auto icp = ColorICP ();
}
