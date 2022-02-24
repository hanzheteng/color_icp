// Color ICP Registration
// Hanzhe Teng, Feb 2022

#include "color_icp/helper.h"
#include "color_icp/remove_nan.h"

#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/search/kdtree.h>
#include <pcl/search/impl/kdtree.hpp> // needed for L2 distance; skip PCL_NO_PRECOMPILE
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>  // needed for L1 distance

#include <Eigen/Geometry>

using PointT = pcl::PointXYZRGB;
using PointCloudPtr = pcl::PointCloud<PointT>::Ptr;

// refer to pcl/registration/impl/icp.hpp and transformation_estimation_svd.hpp
Eigen::Matrix4f ClassicICPRegistration (const PointCloudPtr& source, const PointCloudPtr& target) {
  // initialization
  int iteration = 100;
  float distance_threshold = 0.05; // icp.setMaxCorrespondenceDistance
  int cloud_size = static_cast<int> (source->size());
  Eigen::Matrix4f transformation = Eigen::Matrix4f::Identity();
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
    transformation = Eigen::umeyama (cloud_src, cloud_tgt, false);
    final_transformation = transformation * final_transformation;
    std::cout << "it = " << t << "; cloud size = " << cloud_size << "; idx = " << idx << std::endl;
    std::cout << "current transformation estimation" << std::endl << final_transformation << std::endl;
  }

  return final_transformation;
}


void visualizeRegistration (const PointCloudPtr& source,
                            const PointCloudPtr& source_transformed, 
                            const PointCloudPtr& target) {
  // Add point clouds to the viewer
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

int main(int argc, char** argv){
  // initialization
  std::string src_filename = "../data/cloud_bin_01.pcd";
  std::string tar_filename = "../data/cloud_bin_02.pcd";
  // std::string src_filename = "../data/frag_115.ply";
  // std::string tar_filename = "../data/frag_116.ply";

  pcl::PointCloud<PointT>::Ptr source_cloud (new pcl::PointCloud<PointT>);
  pcl::PointCloud<PointT>::Ptr target_cloud (new pcl::PointCloud<PointT>);
  Eigen::Matrix4f transformation = Eigen::Matrix4f::Identity();

  // load point cloud
  if (my::loadPointCloud<PointT> (src_filename, *source_cloud))
    std::cout << "[source] Loaded " << source_cloud->size() << " points\n";
  else
    return 0;
  if (my::loadPointCloud<PointT> (tar_filename, *target_cloud))
    std::cout << "[target] Loaded " << target_cloud->size() << " points\n";
  else
    return 0;

  // remove NaN points
  std::vector<int> nan_idx;
  pcl::removeNaNFromPointCloudBruteForce (*source_cloud, *source_cloud, nan_idx);
  pcl::removeNaNFromPointCloudBruteForce (*target_cloud, *target_cloud, nan_idx);
  std::cout << "[source] " << source_cloud->size () << " points after removing NaN points\n";
  std::cout << "[target] " << target_cloud->size () << " points after removing NaN points\n";
  pcl::removeNaNRGBFromPointCloud (*source_cloud, *source_cloud, nan_idx);
  pcl::removeNaNRGBFromPointCloud (*target_cloud, *target_cloud, nan_idx);
  std::cout << "[source] " << source_cloud->size () << " points after removing NaN RGB points\n";
  std::cout << "[target] " << target_cloud->size () << " points after removing NaN RGB points\n";

  // downsample
  float voxel_resolution_ = 0.01;
  pcl::VoxelGrid<PointT> sor;
  sor.setLeafSize (voxel_resolution_, voxel_resolution_, voxel_resolution_);
  sor.setInputCloud (source_cloud);
  sor.filter (*source_cloud);
  sor.setInputCloud (target_cloud);
  sor.filter (*target_cloud);
  std::cout << "[source] Downsampled to " << source_cloud->size () << " points\n";
  std::cout << "[target] Downsampled to " << target_cloud->size () << " points\n";

  // // ICP registration in PCL
  // pcl::PointCloud<PointT> registration_output;
  // pcl::IterativeClosestPoint<PointT, PointT> icp;
  // icp.setMaxCorrespondenceDistance (0.05);
  // icp.setTransformationEpsilon (0.000001);
  // icp.setMaximumIterations (100);
  // icp.setInputSource (source_cloud);
  // icp.setInputTarget (target_cloud);
  // icp.align (registration_output);
  // transformation = icp.getFinalTransformation ();
  // std::cout << "Estimated transformation " << std::endl << transformation << std::endl;

  // Classic ICP registration
  transformation = ClassicICPRegistration(source_cloud, target_cloud);
  std::cout << "Estimated transformation " << std::endl << transformation << std::endl;

  // visualization
  pcl::PointCloud<PointT>::Ptr source_cloud_transformed (new pcl::PointCloud<PointT>);
  pcl::transformPointCloud (*source_cloud, *source_cloud_transformed, transformation);
  visualizeRegistration(source_cloud, source_cloud_transformed, target_cloud);
}
