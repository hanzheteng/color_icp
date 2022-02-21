
#ifndef  MY_HELPER_H
#define  MY_HELPER_H

#include <vector>
#include <fstream>
#include <random>
#include <chrono>
#include <unordered_map>
#include <Eigen/Core>
#include <boost/filesystem.hpp>
//#include <filesystem>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/search/kdtree.h>
#include <pcl/kdtree/kdtree_flann.h>

namespace fs = boost::filesystem;
//namespace fs = std::filesystem;  // if using filesystem in C++17


namespace my {

inline double getAngularError(Eigen::Matrix3d R_exp, Eigen::Matrix3d R_est) {
  return std::abs(std::acos(fmin(fmax(((R_exp.transpose() * R_est).trace() - 1) / 2, -1.0), 1.0)));
}


inline double getTranslationError(Eigen::Vector3d t_exp, Eigen::Vector3d t_est) {
  return (t_exp - t_est).norm();
}


Eigen::Matrix3d matrixExp3 (const Eigen::Vector3d& w, float theta){
  Eigen::Matrix3d bw, R;
  bw << 0, -w(2), w(1), w(2), 0, -w(0), -w(1), w(0), 0;
  R << Eigen::Matrix3d::Identity() + std::sin(theta)*bw + (1-std::cos(theta))*bw*bw;
  return R;
}


Eigen::Matrix4d getT (const Eigen::Matrix3d R, const Eigen::Vector3d& t){
  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  T.topLeftCorner(3, 3) = R;
  T.topRightCorner(3, 1) = t;
  return T;
}


Eigen::Matrix4d getT (const Eigen::Vector3d& w, float theta, const Eigen::Vector3d& t){
  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  T.topLeftCorner(3, 3) = matrixExp3(w.normalized(), theta);
  T.topRightCorner(3, 1) = t;
  return T;
}


Eigen::Matrix4d getRandomT (){
  // random seeds
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis_num(-1, 1);
  std::uniform_real_distribution<> dis_angle(0, 2 * M_PI);

  // uniformly generate unit vector (as rotation axis) by equal-area projection
  Eigen::Vector3d w;
  float z = dis_num(gen);
  float alpha = dis_angle(gen);
  w << std::sqrt(1-z*z)*std::cos(alpha), std::sqrt(1-z*z)*std::sin(alpha), z; 

  // generate translation in [-1, 1] cube
  Eigen::Vector3d t;
  t << dis_num(gen), dis_num(gen), dis_num(gen);  
  
  return getT(w, dis_angle(gen), t);
}


void getTiming(std::string note){
  static std::chrono::steady_clock::time_point tic = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point toc = std::chrono::steady_clock::now();
  auto time_elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(toc - tic);
  tic = toc;
  if (!note.empty()) 
    std::cout << "[TIMING] " << note << ": " << time_elapsed.count() / 1000000.0 << " ms\n";
}


template <typename PointT>
void addNoiseToPointCloud(pcl::PointCloud<PointT>& cloud, float noise_std) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<> dis(0, noise_std);
  for (int idx = 0; idx < cloud.points.size(); ++idx){
    cloud[idx].x = cloud[idx].x + static_cast<float> (dis(gen));
    cloud[idx].y = cloud[idx].y + static_cast<float> (dis(gen));
    cloud[idx].z = cloud[idx].z + static_cast<float> (dis(gen));
  }
}


template <typename PointT>
int intersectPointCloud(const typename pcl::PointCloud<PointT>::Ptr& source,
                        const typename pcl::PointCloud<PointT>::Ptr& target,
                        float epsilon) {
  typename pcl::KdTreeFLANN<PointT>::Ptr tree (new pcl::KdTreeFLANN<PointT>);
  tree->setInputCloud (target);
  std::vector<int> index (1);
  std::vector<float> sqr_distance (1);
  int count = 0;
  float sqr_epsilon = epsilon * epsilon;
  for (int idx = 0; idx < source->points.size (); ++idx) {
    tree->nearestKSearch (source->points[idx], 1, index, sqr_distance);
    if (sqr_distance[0] < sqr_epsilon)
      count++;
  }
  return count;
}


int readDirectory(const std::string& folder, std::vector<std::string>& filenames){
  fs::path dir(folder);
  filenames.clear();
  if (fs::exists(dir) && fs::is_directory(dir)){
    std::vector<fs::path> files;
    std::copy(fs::directory_iterator(dir), fs::directory_iterator(), std::back_inserter(files));
    std::sort(files.begin(), files.end());
    for (std::vector<fs::path>::const_iterator it(files.begin()), it_end(files.end()); it != it_end; ++it){
      std::string filename = (*it).string();
      filenames.push_back(filename);
    }
    return 1;
  }
  else {
    std::cout << "The directory " << dir << " does not exist\n";
    return 0;
  }
}


int saveCSV(const std::vector<float>& data, const std::string& filename){
  if (data.empty() || filename.empty())
    return 0;
  fs::path p(filename);
  fs::path dir = p.parent_path();
  if (!dir.empty() && !fs::exists(dir))
    fs::create_directories(dir);

  std::ofstream myfile;
  myfile.open (filename);
  if (myfile.fail())
    return 0;
  for (auto& d : data){
    myfile << std::to_string(d) << std::endl;
  }
  myfile.close ();
  return 1;
}


template <typename PointT>
int loadPointCloud(const std::string& filename, pcl::PointCloud<PointT>& cloud){
  std::string extension = filename.substr(filename.size() - 4, 4);
  if (extension == ".pcd"){
    if (pcl::io::loadPCDFile (filename, cloud) == -1) {
      std::cout << "Was not able to open file " << filename << std::endl;
      return 0;
    }
  }
  else if (extension == ".ply"){
    if (pcl::io::loadPLYFile (filename, cloud) == -1) {
      std::cout << "Was not able to open file " << filename << std::endl;
      return 0;
    }
  }
  else {
    std::cerr << "Was not able to open file " << filename
              << " (it is neither .pcd nor .ply) " << std::endl;
    return 0;
  }
  return 1;
}


struct FramedTransformation {
  int id1_;
  int id2_;
  int frame_;
  Eigen::Matrix4d transformation_;
  FramedTransformation( int id1, int id2, int f, Eigen::Matrix4d t )
    : id1_( id1 ), id2_( id2 ), frame_( f ), transformation_( t )
    {}
};


struct Trajectory {
  std::vector< FramedTransformation > data_;
  int index_;

  void loadFromFile( std::string filename ) {
    data_.clear();
    index_ = 0;
    int id1, id2, frame;
    Eigen::Matrix4d trans;
    FILE * f = fopen( filename.c_str(), "r" );
    if ( f != NULL ) {
      char buffer[1024];
      while ( fgets( buffer, 1024, f ) != NULL ) {
        if ( strlen( buffer ) > 0 && buffer[ 0 ] != '#' ) {
          sscanf( buffer, "%d %d %d", &id1, &id2, &frame);
          fgets( buffer, 1024, f );
          sscanf( buffer, "%lf %lf %lf %lf", &trans(0,0), &trans(0,1), &trans(0,2), &trans(0,3) );
          fgets( buffer, 1024, f );
          sscanf( buffer, "%lf %lf %lf %lf", &trans(1,0), &trans(1,1), &trans(1,2), &trans(1,3) );
          fgets( buffer, 1024, f );
          sscanf( buffer, "%lf %lf %lf %lf", &trans(2,0), &trans(2,1), &trans(2,2), &trans(2,3) );
          fgets( buffer, 1024, f );
          sscanf( buffer, "%lf %lf %lf %lf", &trans(3,0), &trans(3,1), &trans(3,2), &trans(3,3) );
          data_.push_back( FramedTransformation( id1, id2, frame, trans ) );
        }
      }
      fclose( f );
    }
  }
  void saveToFile( std::string filename ) {
    FILE * f = fopen( filename.c_str(), "w" );
    for ( int i = 0; i < ( int )data_.size(); i++ ) {
      Eigen::Matrix4d & trans = data_[ i ].transformation_;
      fprintf( f, "%d\t%d\t%d\n", data_[ i ].id1_, data_[ i ].id2_, data_[ i ].frame_ );
      fprintf( f, "%.8f %.8f %.8f %.8f\n", trans(0,0), trans(0,1), trans(0,2), trans(0,3) );
      fprintf( f, "%.8f %.8f %.8f %.8f\n", trans(1,0), trans(1,1), trans(1,2), trans(1,3) );
      fprintf( f, "%.8f %.8f %.8f %.8f\n", trans(2,0), trans(2,1), trans(2,2), trans(2,3) );
      fprintf( f, "%.8f %.8f %.8f %.8f\n", trans(3,0), trans(3,1), trans(3,2), trans(3,3) );
    }
    fclose( f );
  }
};

} // namespace my

#endif  //#ifndef MY_HELPER_H_

