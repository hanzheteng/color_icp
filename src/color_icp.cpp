
#include "color_icp/helper.h"
#include "color_icp/remove_nan.h"

#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/search/kdtree.h>
#include <pcl/search/impl/kdtree.hpp> // needed for L2 distance; skip PCL_NO_PRECOMPILE
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>  // needed for L1 distance


using PointT = pcl::PointXYZRGB;
using PointCloudPtr = pcl::PointCloud<PointT>::Ptr;

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

  // ICP registration
  pcl::PointCloud<PointT> registration_output;
  pcl::IterativeClosestPoint<PointT, PointT> icp;
  icp.setMaxCorrespondenceDistance (0.05);
  icp.setTransformationEpsilon (0.000001);
  icp.setMaximumIterations (100);
  icp.setInputSource (source_cloud);
  icp.setInputTarget (target_cloud);
  icp.align (registration_output);
  transformation = icp.getFinalTransformation ();
  std::cout << "Estimated transformation " << std::endl << transformation << std::endl;

  // visualization
  pcl::PointCloud<PointT>::Ptr source_cloud_transformed (new pcl::PointCloud<PointT>);
  pcl::transformPointCloud (*source_cloud, *source_cloud_transformed, transformation);
  visualizeRegistration(source_cloud, source_cloud_transformed, target_cloud);
}
