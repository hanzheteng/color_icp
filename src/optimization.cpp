// Color ICP Registration
// Hanzhe Teng, Feb 2022

#include <vector>
#include <iostream>
#include <random>
#include <Eigen/Core>
#include <Eigen/Geometry>


// curve fitting: f = exp(ax^2 + bx + c)
Eigen::Matrix2Xf dataGeneration () {
  float a = 1, b = 2, c = 3; // ground truth
  auto func = [=](float x)->float {
    return std::exp(a*x*x + b*x + c);
  };
  Eigen::Matrix2Xf data (2, 100);
  float noise_std = 1;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<> dis(0, noise_std);
  for (int i = 0; i < 100; ++i) {
    data (0, i) = i * 0.01;
    data (1, i) = func (i*0.01) + static_cast<float> (dis(gen));
  }
  return data;
}


Eigen::Vector3f GaussNewton (Eigen::Matrix2Xf& data, Eigen::Vector3f& guess) {
  // model: f = exp(a*x*x + b*x + c);
  int size = 100;
  Eigen::Vector3f var = guess;
  for (int t = 0; t < 10; ++t) {
    float a = var(0);
    float b = var(1);
    float c = var(2);
    Eigen::Matrix3Xf Jacobian (3, size);
    Eigen::VectorXf Residual (size);
    for (int i = 0; i < size; ++i) {
      float x = data(0, i);
      Residual (i) = data(1, i) - std::exp(a*x*x + b*x + c);
      Jacobian(0, i) = x*x * std::exp(a*x*x + b*x + c);
      Jacobian(1, i) = x * std::exp(a*x*x + b*x + c);
      Jacobian(2, i) = std::exp(a*x*x + b*x + c);
    }
    var += (Jacobian * Jacobian.transpose()).inverse() * Jacobian * Residual;
    std::cout << "current var = " << var << std::endl;
  }
  return var;
}


int main(int argc, char** argv){
  // initialization
  auto data = dataGeneration();
  std::cout << "data = " << std::endl << data << std::endl;

  // solve by non-linear optimization
  Eigen::Vector3f guess (0.5, 2.2, 3.4); // a, b, c
  auto result = GaussNewton (data, guess);
  std::cout << "final result = " << result << std::endl;
}
