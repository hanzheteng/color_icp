# color_icp

## Introduction

This repository serves as a learning material for ICP and Colored ICP algorithms. The code is well organized and clean. We aim to focus on only the main pipeline of the algorithm, and avoid complicated interfaces and nested templates as in large libraries such as PCL and Open3D. 

- The ICP algorithm follows a typical pipeline, and our implementation herein can achieve exactly the same behavior/performance as in PCL.
- The Colored ICP algorithm is an implementation of the paper: Colored Point Cloud Registration Revisited, Jaesik Park, Qian-Yi Zhou and Vladlen Koltun, ICCV 2017
- The original/official implementation of the Colored ICP algorithm is [available at Open3D Github Repository](https://github.com/isl-org/Open3D/blob/master/cpp/open3d/pipelines/registration/ColoredICP.cpp). Part of the implementation in this repository has been merged into the Open3D library (See [Open3D PR#4988](https://github.com/isl-org/Open3D/pull/4988)).
- We provide some notes to discuss the math used in the Colored ICP algorithm, in particular Residuals and Jacobian matrices. 

## Code Structure

- `config` folder 
  - `params.yaml` The YAML file to control the running flow of the point cloud registration process. We adopt a header-only library [`mini-yaml`](https://github.com/jimmiebergmann/mini-yaml) in this project. It is convenient for tuning parameters without the need of re-compilation of the C++ program.
- `data` folder 
  - Contain a few sample point clouds from [Redwood Synthetic](http://redwood-data.org/indoor/) and [Redwood Scan](http://redwood-data.org/indoor_lidar_rgbd/) datasets.
- `include` folder
  - `color_icp/helper.h` Provide some helper functions that were developed in some other projects of mine. Only the `loadPointCloud` function is used in this project. Feel free to make use of the rest of helper functions as you see fit.
  - `color_icp/remove_nan.h` Include some customized functions to remove NaN points in the point cloud; they are modified from PCL.
  - `color_icp/yaml.h` The header file adopted from the mini-yaml library.
- `scripts` folder
  - `colored_icp.py` A python script that runs ICP and Colored ICP algorithms using the API provided by Open3D. It can be used to compare the performance of our code with that of Open3D.
- `src` folder
  - `color_icp.cpp` The core implementation of the registration pipeline. It takes in the `params.yaml` file and runs modular-designed functions accordingly.
  - `optimization.cpp` A simple practice code to solve a curve fitting problem using Gauss-Newton method.
  - `yaml.cpp`  The cpp file adopted from the mini-yaml library.
- `Notes_on_Colored_Point_Cloud_Registration.pdf` Some math notes about residuals and Jacobian matrices used in the Colored ICP algorithm.


## Build and Run

The code was developed under Ubuntu 18. When needed, PCL 1.8 is used (the default version under Ubuntu 18). We follow a typical compilation procedure using CMake. 

```
mkdir build
cd build
cmake ..
make
```

The running flow is controlled via the `params.yaml` file under the `config` folder. Modify the YAML file as you like, and run

```
./color_icp
```

## Debugging
- During my experiments, `float` precision was not good enough and can cause numerical instability at convergence. This can be observed in the JTJ and JTr matrices. Switching to `double` precision solved this issue.

