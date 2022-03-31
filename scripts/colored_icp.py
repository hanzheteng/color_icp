#!/usr/bin/env python3
import os
import copy
import numpy as np
import open3d as o3d


def draw_registration_result(source, target, transformation, color=False):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    if not color:
        source_temp.paint_uniform_color([1, 1, 0])
        target_temp.paint_uniform_color([0, 1, 1])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def icp_registration(source, target, max_corres_dist):
    print("ICP registration")
    result = o3d.pipelines.registration.registration_icp(source, target, max_corres_dist)
    print(result)
    print(result.transformation)
    draw_registration_result(source, target, result.transformation, color=False)


def colored_icp_registration(source, target, voxel_size):
    print("Colored ICP registration")
    voxel_radius = [5*voxel_size, 3*voxel_size, voxel_size]
    max_iter = [60, 35, 20]
    current_transformation = np.identity(4)
    for scale in range(3):
        max_it = max_iter[scale]
        radius = voxel_radius[scale]
        print("scale_level = {0}, voxel_size = {1}, max_iter = {2}".format(scale, radius, max_it))
        source_down = source.voxel_down_sample(radius)
        target_down = target.voxel_down_sample(radius)
        source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius*2, max_nn=20))
        target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius*2, max_nn=20))
        result = o3d.pipelines.registration.registration_colored_icp(
            source_down, 
            target_down, 
            radius, 
            current_transformation,
            o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                              relative_rmse=1e-6,
                                                              max_iteration=max_it))
        current_transformation = result.transformation
        print(result)
    print(current_transformation)
    draw_registration_result(source, target, current_transformation, color=False)


if __name__ == "__main__":
    # unset env variable to re-enable OpenGL 3.3 in VM environments
    if 'SVGA_VGPU10' in os.environ:
        del os.environ['SVGA_VGPU10']

    # parameters
    voxel_size = 0.01
    max_corres_dist = 5*voxel_size
    transformation = np.identity(4)
    # trans_init = np.asarray([[0.862, 0.011, -0.507, 0.5],
    #                         [-0.139, 0.967, -0.215, 0.7],
    #                         [0.487, 0.255, 0.835, -1.4], [0.0, 0.0, 0.0, 1.0]])

    # load point clouds
    source = o3d.io.read_point_cloud("../data/cloud_bin_00_easy.pcd")
    target = o3d.io.read_point_cloud("../data/cloud_bin_01.pcd")
    # source = o3d.io.read_point_cloud("../data/frag_115.ply")
    # target = o3d.io.read_point_cloud("../data/frag_116.ply")
    print("Loaded " + str(len(source.points)) + " points for source point cloud")
    print("Loaded " + str(len(target.points)) + " points for target point cloud")
    
    # estimate normals
    source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2*voxel_size, max_nn=16))
    target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2*voxel_size, max_nn=16))
    print("Estimated normals for source and target point clouds")

    # show initial alignment
    print("Initial alignment")
    print(o3d.pipelines.registration.evaluate_registration(source, target, max_corres_dist))
    draw_registration_result(source, target, transformation, color=False)

    # ICP
    icp_registration(source, target, max_corres_dist)

    # colored ICP
    colored_icp_registration(source, target, voxel_size)
