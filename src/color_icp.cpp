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
    typedef Eigen::Matrix<double, 6, 6> Matrix6d;
    typedef Eigen::Matrix<double, 6, 1> Vector6d;
    typedef Eigen::Matrix<double, 6, -1> Matrix6Xd;
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
    void visualizeRegistrationWithColor (const PointCloudPtr& source_transformed, const PointCloudPtr& target);
    void prepareColorGradient (const PointCloudPtr& target);

    Eigen::Matrix4d PCLICP (const PointCloudPtr& source, const PointCloudPtr& target);
    Eigen::Matrix4d ClassicICPRegistration (const PointCloudPtr& source, const PointCloudPtr& target);
    Eigen::Matrix4d ColorICPRegistration (const PointCloudPtr& source, const PointCloudPtr& target);
    Eigen::Matrix4d GaussNewton (const PointCloudPtr& source, const PointCloudPtr& target);
    Eigen::Matrix4d GaussNewtonWithColor (const PointCloudPtr& source, const PointCloudPtr& target, 
                                          const std::vector<Eigen::Vector3d>& target_gradient);
    Eigen::Matrix4d TransformVector6dToMatrix4d (const Eigen::Vector6d& input);

  private:
    pcl::search::KdTree<PointT, pcl::KdTreeFLANN<PointT, flann::L2_Simple<float>>>::Ptr kdtree_;
    pcl::PointCloud<PointT>::Ptr source_cloud_;
    pcl::PointCloud<PointT>::Ptr target_cloud_;
    std::vector<Eigen::Vector3d> target_color_gradient_;
    Eigen::Vector3d rgb_to_intensity_weight_;
    Eigen::Matrix4d transformation_;
    Yaml::Node params_;
};


ColorICP::ColorICP ()
    : source_cloud_ (new pcl::PointCloud<PointT>)
    , target_cloud_ (new pcl::PointCloud<PointT>)
    , transformation_ (Eigen::Matrix4d::Identity ())
    , rgb_to_intensity_weight_ (Eigen::Vector3d::Ones () / 3.0)
    , kdtree_ (new pcl::search::KdTree<PointT, pcl::KdTreeFLANN<PointT, flann::L2_Simple<float>>>) {

  // init params
  Yaml::Parse (params_, "../config/params.yaml");
  std::string color_weight = params_["color_weight"].As<std::string> ();
  if (color_weight == "ntsc")
    rgb_to_intensity_weight_ = Eigen::Vector3d (0.299, 0.587, 0.114); // YIQ, YUV and NTSC
  else if (color_weight == "srgb")
    rgb_to_intensity_weight_ = Eigen::Vector3d (0.2126, 0.7152, 0.0722); // sRGB (and Rec709)
  // see paper: Why You Should Forget Luminance Conversion and Do Something Better, CVPR 2017

  // load point clouds
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
  if (params_["color_visualization"].As<bool> ())
    visualizeRegistrationWithColor(source_cloud_transformed, target_cloud_);
  else
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
  double resolution = params_["voxel_resolution"].As<double> ();
  pcl::VoxelGrid<PointT> sor;
  sor.setLeafSize (resolution, resolution, resolution);
  sor.setInputCloud (cloud);
  sor.filter (*cloud);
  std::cout << "Downsampled to " << cloud->size () << " points\n";
}


void ColorICP::estimateNormals (const PointCloudPtr& cloud) {
  pcl::NormalEstimation<PointT, NormalT> ne;
  ne.setSearchMethod (kdtree_);
  ne.setRadiusSearch (params_["normal_est_radius"].As<double> ());
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


void ColorICP::visualizeRegistrationWithColor (const PointCloudPtr& source_transformed, 
                                               const PointCloudPtr& target) {
  // add point clouds to the viewer
  pcl::visualization::PCLVisualizer visualizer;
  visualizer.addPointCloud<PointT> (source_transformed, "source cloud transformed");
  visualizer.addPointCloud<PointT> (target, "target cloud");
  
  while (!visualizer.wasStopped ()) {
    visualizer.spinOnce ();
    pcl_sleep(0.01);
  }
}


Eigen::Matrix4d ColorICP::PCLICP (const PointCloudPtr& source, const PointCloudPtr& target) {
  pcl::PointCloud<PointT> registration_output;
  pcl::IterativeClosestPoint<PointT, PointT, double> icp;
  icp.setMaxCorrespondenceDistance (params_["icp_max_corres_dist"].As<double> ());
  icp.setTransformationEpsilon (params_["icp_transformation_epsilon"].As<double> ());
  icp.setMaximumIterations (params_["icp_max_iterations"].As<int> ());
  icp.setInputSource (source_cloud_);
  icp.setInputTarget (target_cloud_);
  icp.align (registration_output);
  return icp.getFinalTransformation ();
}


// refer to pcl/registration/impl/icp.hpp and transformation_estimation_svd.hpp
// exactly the same behavior and performance as using PCLICP()
Eigen::Matrix4d ColorICP::ClassicICPRegistration (const PointCloudPtr& source, const PointCloudPtr& target) {
  // initialization
  int iteration = params_["icp_max_iterations"].As<int> ();
  double translation_epsilon = params_["icp_transformation_epsilon"].As<double> ();
  double rotation_epsilon = 1.0 - params_["icp_transformation_epsilon"].As<double> ();
  double distance_threshold = params_["icp_max_corres_dist"].As<double> (); // icp.setMaxCorrespondenceDistance
  int cloud_size = static_cast<int> (source->size());
  Eigen::Matrix4d final_transformation = Eigen::Matrix4d::Identity();
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
    Eigen::Matrix3Xd cloud_src (3, correspondences.size());
    Eigen::Matrix3Xd cloud_tgt (3, correspondences.size());
    for (const auto& corres : correspondences) {
      cloud_src (0, idx) = source_trans->points[corres.first].x;
      cloud_src (1, idx) = source_trans->points[corres.first].y;
      cloud_src (2, idx) = source_trans->points[corres.first].z;
      cloud_tgt (0, idx) = target->points[corres.second].x;
      cloud_tgt (1, idx) = target->points[corres.second].y;
      cloud_tgt (2, idx) = target->points[corres.second].z;
      ++idx;
    }

    // skip a few steps (such as sanity checks) here for simplicity

    // solve using Umeyama's algorithm (SVD)
    Eigen::Matrix4d transformation = Eigen::umeyama<Eigen::Matrix3Xd, Eigen::Matrix3Xd> (cloud_src, cloud_tgt, false);
    final_transformation = transformation * final_transformation;
    std::cout << "it = " << t << "; cloud size = " << cloud_size << "; idx = " << idx << std::endl;
    std::cout << "current transformation estimation" << std::endl << final_transformation << std::endl;

    // check convergence
    double cos_angle = 0.5 * (transformation.coeff (0, 0) + transformation.coeff (1, 1) + transformation.coeff (2, 2) - 1);
    double translation_sqr = transformation.coeff (0, 3) * transformation.coeff (0, 3) +
                             transformation.coeff (1, 3) * transformation.coeff (1, 3) +
                             transformation.coeff (2, 3) * transformation.coeff (2, 3);
    if (cos_angle >= rotation_epsilon && translation_sqr <= translation_epsilon) {
      std::cout << "converged!" << std::endl;
      break;
    }
  }

  return final_transformation;
}


// Implementation according to the paper: Colored Point Cloud Registration Revisited, ICCV 2017
// https://github.com/isl-org/Open3D/blob/master/cpp/open3d/pipelines/registration/ColoredICP.cpp
// https://github.com/isl-org/Open3D/blob/master/cpp/open3d/utility/Eigen.cpp
Eigen::Matrix4d ColorICP::ColorICPRegistration (const PointCloudPtr& source, const PointCloudPtr& target) {
  // initialization
  int iteration = params_["icp_max_iterations"].As<int> ();
  double translation_epsilon = params_["icp_transformation_epsilon"].As<double> ();
  double rotation_epsilon = 1.0 - params_["icp_transformation_epsilon"].As<double> ();
  double distance_threshold = params_["icp_max_corres_dist"].As<double> (); // icp.setMaxCorrespondenceDistance
  int cloud_size = static_cast<int> (source->size());
  Eigen::Matrix4d final_transformation = Eigen::Matrix4d::Identity();
  pcl::PointCloud<PointT>::Ptr source_trans (new pcl::PointCloud<PointT>);
  prepareColorGradient(target);

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
    int corres_size = static_cast<int> (indices_src.size());
    std::cout << "it = " << t << "; cloud size = " << cloud_size << "; selected size = " << corres_size << std::endl;

    // copy selected correspondences to new point clouds
    pcl::PointCloud<PointT>::Ptr cloud_src (new pcl::PointCloud<PointT>);
    pcl::PointCloud<PointT>::Ptr cloud_tgt (new pcl::PointCloud<PointT>);
    copyPointCloud (source_trans, indices_src, cloud_src);
    copyPointCloud (target, indices_tgt, cloud_tgt);

    // copy color gradient accordingly for selected correspondences
    std::vector<Eigen::Vector3d> gradient_tgt (corres_size);
    for (int i = 0; i < corres_size; ++i) {
      gradient_tgt[i] = target_color_gradient_[indices_tgt[i]];
    }

    // solve using Gauss-Newton method
    Eigen::Matrix4d transformation = GaussNewtonWithColor (cloud_src, cloud_tgt, gradient_tgt);
    final_transformation = transformation * final_transformation;
    std::cout << "transformation in this iteration" << std::endl << transformation << std::endl;
    std::cout << "current transformation estimation" << std::endl << final_transformation << std::endl;

    // check convergence
    double cos_angle = 0.5 * (transformation.coeff (0, 0) + transformation.coeff (1, 1) + transformation.coeff (2, 2) - 1);
    double translation_sqr = transformation.coeff (0, 3) * transformation.coeff (0, 3) +
                             transformation.coeff (1, 3) * transformation.coeff (1, 3) +
                             transformation.coeff (2, 3) * transformation.coeff (2, 3);
    if (cos_angle >= rotation_epsilon && translation_sqr <= translation_epsilon) {
      std::cout << "converged!" << std::endl;
      break;
    }
  }

  return final_transformation;
}


// refer to InitializePointCloudForColoredICP() in open3d/pipelines/registration/ColoredICP.cpp 
void ColorICP::prepareColorGradient (const PointCloudPtr& target) {
  int cloud_size = static_cast<int> (target->size());
  pcl::search::KdTree<PointT>::Ptr kdtree (new pcl::search::KdTree<PointT>);
  kdtree->setInputCloud(target);
  std::vector<int> indices (1);
  std::vector<float> sq_dist (1);
  double search_radius = params_["search_radius"].As<double> ();
  target_color_gradient_.clear();
  target_color_gradient_.resize(cloud_size, Eigen::Vector3d::Zero());

  // find color gradient for each point in the target cloud
  for (int i = 0; i < cloud_size; ++i) {
    const Eigen::Vector3d p = target->points[i].getVector3fMap().cast<double>();
    const Eigen::Vector3d np = target->points[i].getNormalVector3fMap().cast<double>();
    const Eigen::Vector3d cp = target->points[i].getRGBVector3i().cast<double>();
    double ip = cp.dot(rgb_to_intensity_weight_) / 255.0;
    // TODO: test varying weights to combine RGB --> less sensitive; almost no influence; depends on the scene
    // TODO: test if the magnitude of intensity matters in optimization --> a lot; has to be [0, 1]

    // search neighbor points for each point in the cloud
    kdtree->radiusSearch(target->points[i], search_radius, indices, sq_dist);
    int nn_size = static_cast<int> (indices.size());
    if (nn_size < 5) {
      std::cerr << "[WARNING]: not enough points in the neighborhood; size = " << nn_size << "; skipping ... \n";
      continue; // TODO: test the influence of leaving color gradient zero --> no contribution from this component
    }

    // solve a linear least-squares problem to find the color gradient; see Eq. 10 in the paper
    Eigen::Matrix3Xd A (3, nn_size); // we have transposed A here for the convenience of computation
    Eigen::VectorXd b (nn_size);
    for (int k = 1; k < nn_size; ++k) { // skip index 0, which is the query point itself
      const Eigen::Vector3d pp = target->points[indices[k]].getVector3fMap().cast<double>();
      const Eigen::Vector3d cpp = target->points[indices[k]].getRGBVector3i().cast<double>();
      double ipp = cpp.dot(rgb_to_intensity_weight_) / 255.0;

      // project neighbor points to p's tangent plane
      const Eigen::Vector3d& pp_proj = pp - (pp-p).dot(np)*np;
      A.block<3, 1>(0, k-1) = pp_proj - p;
      b (k-1) = ipp - ip;
    }
    // add orthogonal constraint dp.dot(np) = 0 with weight (nn_size-1)
    A.block<3, 1>(0, nn_size-1) = (nn_size-1) * np;
    b (nn_size-1) = 0; // TODO: test the influence of this constraint --> can make optimization tighter

    // solve for dp
    Eigen::Vector3d X = (A * A.transpose()).ldlt().solve(A * b);
    target_color_gradient_[i] = X;  // X is the estimated color gradient dp
  }
  std::cout << "Completed color gradient estimation for target cloud\n";
}


// refer to TransformationEstimationForColoredICP::ComputeTransformation in open3d/pipelines/registration/ColoredICP.cpp
Eigen::Matrix4d ColorICP::GaussNewtonWithColor (const PointCloudPtr& source, const PointCloudPtr& target, 
                                                const std::vector<Eigen::Vector3d>& target_gradient) {
  int size = static_cast<int> (source->size());
  Eigen::Matrix6Xd JacobianGeo (6, size);
  Eigen::VectorXd ResidualGeo (size);
  Eigen::Matrix6Xd JacobianColor (6, size);
  Eigen::VectorXd ResidualColor (size);
  double lambda = params_["color_icp_lambda"].As<double> (); // TODO: test varying param values

  for (int i = 0; i < size; ++i) {
    const Eigen::Vector3d q = source->points[i].getVector3fMap().cast<double>();
    const Eigen::Vector3d cq = source->points[i].getRGBVector3i().cast<double>();
    double iq = cq.dot(rgb_to_intensity_weight_) / 255.0;
    
    const Eigen::Vector3d p = target->points[i].getVector3fMap().cast<double>();
    const Eigen::Vector3d np = target->points[i].getNormalVector3fMap().cast<double>();
    const Eigen::Vector3d cp = target->points[i].getRGBVector3i().cast<double>();
    double ip = cp.dot(rgb_to_intensity_weight_) / 255.0;
    const Eigen::Vector3d& dp = target_gradient[i];

    ResidualGeo (i) = (q - p).dot(np);           // Eq. 19 in the paper
    JacobianGeo.block<3, 1>(0, i) = q.cross(np); // Eq. 30 in the paper
    JacobianGeo.block<3, 1>(3, i) = np;

    const Eigen::Vector3d& q_proj = q - (q-p).dot(np) * np;
    const double& iq_proj = ip + dp.dot(q_proj-p);
    const Eigen::Matrix3d& M = Eigen::Matrix3d::Identity() - np * np.transpose();
    const Eigen::Vector3d& dpM = dp.transpose() * M;

    ResidualColor (i) = iq_proj - iq;               // Eq. 18 in the paper
    JacobianColor.block<3, 1>(0, i) = q.cross(dpM); // Eq. 28-29 in the paper
    JacobianColor.block<3, 1>(3, i) = dpM;
  }

  Eigen::Matrix6d JTJ_G = JacobianGeo * JacobianGeo.transpose();
  Eigen::Vector6d JTr_G = JacobianGeo * ResidualGeo;
  Eigen::Matrix6d JTJ_C = JacobianColor * JacobianColor.transpose();
  Eigen::Vector6d JTr_C = JacobianColor * ResidualColor;

  Eigen::Matrix6d JTJ = sqrt(lambda) * JTJ_G + sqrt(1-lambda) * JTJ_C;
  Eigen::Vector6d JTr = sqrt(lambda) * JTr_G + sqrt(1-lambda) * JTr_C;
  Eigen::Vector6d X = JTJ.ldlt().solve(-JTr);
  std::cout << "JTJ = \n" <<  JTJ << std::endl;
  std::cout << "JTr = \n" <<  JTr << std::endl;
  std::cout << "X = \n" <<  X << std::endl;
  return TransformVector6dToMatrix4d(X);
}


Eigen::Matrix4d ColorICP::GaussNewton (const PointCloudPtr& source, const PointCloudPtr& target) {
  int size = static_cast<int> (source->size());
  Eigen::Matrix6Xd Jacobian (6, size); // alpha, beta, gamma, a, b, c as in the linearized transformation matrix
  Eigen::VectorXd Residual (size);     // see Eq. 20 in the paper

  for (int i = 0; i < size; ++i) {
    const Eigen::Vector3d q = source->points[i].getVector3fMap().cast<double>();
    const Eigen::Vector3d p = target->points[i].getVector3fMap().cast<double>();
    const Eigen::Vector3d np = target->points[i].getNormalVector3fMap().cast<double>();
    Residual (i) = (q - p).dot(np);           // Eq. 19 in the paper
    Jacobian.block<3, 1>(0, i) = q.cross(np); // Eq. 30 in the paper
    Jacobian.block<3, 1>(3, i) = np;
  }

  Eigen::Matrix6d JTJ = Jacobian * Jacobian.transpose(); // Jacobian herein has already been transposed (row vector)
  Eigen::Vector6d JTr = Jacobian * Residual;
  Eigen::Vector6d X = JTJ.ldlt().solve(-JTr); // solve a system of linear equations in the form: AX = b
  return TransformVector6dToMatrix4d(X);
}


Eigen::Matrix4d ColorICP::TransformVector6dToMatrix4d(const Eigen::Vector6d& input) {
  Eigen::Matrix4d output;
  output.setIdentity();
  // AngleAxis representation implicitly maps the linearized matrix to SO(3)
  output.block<3, 3>(0, 0) = (Eigen::AngleAxisd(input(2), Eigen::Vector3d::UnitZ()) *
                              Eigen::AngleAxisd(input(1), Eigen::Vector3d::UnitY()) *
                              Eigen::AngleAxisd(input(0), Eigen::Vector3d::UnitX())).matrix();
  output.block<3, 1>(0, 3) = input.block<3, 1>(3, 0);
  return output;
}


int main(int argc, char** argv) {
  auto icp = ColorICP ();
}
