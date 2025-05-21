# This file is part of a project that includes code adapted from BlenderProc (GPL-3.0).
#
# Copyright (C) 2021 German Aerospace Center (DLR)
# Modified by  Sojeong Kim, 2025
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import blenderproc as bproc
import argparse
import os
import numpy as np

# argument setting
custom_path = 'custom_set/'
custom_name = 'custom'
cc_textures_path = 'resources/cctextures'

parser = argparse.ArgumentParser()
parser.add_argument('bop_parent_path', help="Path to the bop datasets parent directory")
parser.add_argument('output_dir', help="Path to where the final files will be saved ")
parser.add_argument('--obj_id', type=int, required=True, help="Target object ID (e.g., 1 for obj_000001.ply)")
parser.add_argument('--num_scenes', type=int, default=1, help="Number of scenes (e.g., 400 for total 10000 images)")
args = parser.parse_args()

bproc.init()

# load custom obj into the scene
target_custom_objs = bproc.loader.load_bop_objs(
                    bop_dataset_path=os.path.join(custom_path, custom_name),
                    obj_ids=[args.obj_id],
                    mm2m=True,
                    sample_objects=False,
                    num_of_objs_to_sample=1
)

# load distractor bop objects
tless_dist_bop_objs = bproc.loader.load_bop_objs(bop_dataset_path = os.path.join(args.bop_parent_path, 'tless'), model_type = 'cad', mm2m = True)
lm_dist_bop_objs = bproc.loader.load_bop_objs(bop_dataset_path = os.path.join(args.bop_parent_path, 'lm'), mm2m = True)

# load BOP datset intrinsics
bproc.loader.load_bop_intrinsics(bop_dataset_path = os.path.join(custom_path, custom_name), type="custom")

# set shading and hide objects
for obj in (target_custom_objs + tless_dist_bop_objs + lm_dist_bop_objs):
    obj.set_shading_mode('auto')    
    obj.hide(True)

# create room (5 planes)
room_planes = [bproc.object.create_primitive('PLANE', scale=[2, 2, 1]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, -2, 2], rotation=[-1.570796, 0, 0]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, 2, 2], rotation=[1.570796, 0, 0]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[2, 0, 2], rotation=[0, -1.570796, 0]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[-2, 0, 2], rotation=[0, 1.570796, 0])]

# sample light color and strenght from ceiling
light_plane = bproc.object.create_primitive('PLANE', scale=[3, 3, 1], location=[0, 0, 10])
light_plane.set_name('light_plane')
light_plane_material = bproc.material.create('light_material')

# sample point light on shell
light_point = bproc.types.Light()
light_point.set_energy(200)

# load cc_textures
cc_textures = bproc.loader.load_ccmaterials(cc_textures_path)

# Define a function that samples 6-DoF poses (Object)
def sample_pose_func(obj: bproc.types.MeshObject):
    min = np.random.uniform([-0.3, -0.3, 0.0], [-0.2, -0.2, 0.0])
    max = np.random.uniform([0.2, 0.2, 0.4], [0.3, 0.3, 0.6])
    obj.set_location(np.random.uniform(min, max))
    obj.set_rotation_euler(bproc.sampler.uniformSO3())
    
# activate depth rendering without antialiasing and set amount of samples for color rendering
bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.renderer.set_max_amount_of_samples(50)

print(f"Generating images for object {args.obj_id}...")

images_per_scene = 25
num_scenes = args.num_scenes  # 400 scenes
num_total_images = images_per_scene * num_scenes   # 10000

for i in range(num_scenes):

    # Sample bop objects for a scene
    sampled_distractor_bop_objs = list(np.random.choice(tless_dist_bop_objs, size=4, replace=False))
    sampled_distractor_bop_objs += list(np.random.choice(lm_dist_bop_objs, size=5, replace=False))


    # Randomize materials and set physics
    for obj in (target_custom_objs + sampled_distractor_bop_objs):        
        mat = obj.get_materials()[0]
        if obj.get_cp("bop_dataset_name") in ['itodd', 'tless']:
            grey_col = np.random.uniform(0.1, 0.9)   
            mat.set_principled_shader_value("Base Color", [grey_col, grey_col, grey_col, 1])        
        mat.set_principled_shader_value("Roughness", np.random.uniform(0, 1.0))
        mat.set_principled_shader_value("Specular IOR Level", np.random.uniform(0, 1.0))
        obj.hide(False)
    
    # Sample two light sources
    light_plane_material.make_emissive(emission_strength=np.random.uniform(3,6), 
                                    emission_color=np.random.uniform([0.5, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0]))  
    light_plane.replace_materials(light_plane_material)
    light_point.set_color(np.random.uniform([0.5,0.5,0.5],[1,1,1]))
    location = bproc.sampler.shell(center = [0, 0, 0], radius_min = 1, radius_max = 1.5,
                            elevation_min = 5, elevation_max = 89)
    light_point.set_location(location)

    # sample CC Texture and assign to room planes
    random_cc_texture = np.random.choice(cc_textures)
    for plane in room_planes:
        plane.replace_materials(random_cc_texture)

    # Sample object poses and check collisions 
    bproc.object.sample_poses(objects_to_sample = target_custom_objs + sampled_distractor_bop_objs,
                            sample_pose_func = sample_pose_func, 
                            max_tries = 1000)
            
    # Define a function that samples the initial pose of a given object above the ground
    def sample_initial_pose(obj: bproc.types.MeshObject):
        obj.set_location(bproc.sampler.upper_region(objects_to_sample_on=room_planes[0:1],
                                                    min_height=1, max_height=4, face_sample_range=[0.4, 0.6]))
        obj.set_rotation_euler(np.random.uniform([0, 0, 0], [0, 0, np.pi * 2]))

    # Sample objects on the given surface
    placed_objects = bproc.object.sample_poses_on_surface(objects_to_sample=target_custom_objs + sampled_distractor_bop_objs,
                                                          surface=room_planes[0],
                                                          sample_pose_func=sample_initial_pose,
                                                          min_distance=0.01,
                                                          max_distance=0.2)

    # BVH tree used for camera obstacle checks
    bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects(target_custom_objs + sampled_distractor_bop_objs)

    cam_poses = 0
    while cam_poses < images_per_scene:
        # Sample location
        location = bproc.sampler.shell(center = [0, 0, 0],
                                radius_min = 0.6,
                                radius_max = 1.2,
                                elevation_min = 5,
                                elevation_max = 89)
        
        # Determine point of interest in scene as the object closest to the mean of a subset of objects
        poi = bproc.object.compute_poi(target_custom_objs)
        # Compute rotation based on vector going from location towards poi
        rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=np.random.uniform(-0.7854, 0.7854))
        # Add homog cam pose based on location an rotation
        cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
        
        # Check that obstacles are at least 0.3 meter away from the camera and make sure the view interesting enough
        if bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, {"min": 0.3}, bop_bvh_tree):
            # Persist camera pose
            bproc.camera.add_camera_pose(cam2world_matrix, frame=cam_poses)
            cam_poses += 1

    # render the whole pipeline
    data = bproc.renderer.render()

    # Write data in bop format
    bproc.writer.write_bop(os.path.join(args.output_dir, 'bop_data'),
                           target_objects = target_custom_objs,
                           dataset = 'custom',
                           depth_scale = 0.1,
                           depths = data["depth"],
                           colors = data["colors"], 
                           color_file_format = "JPEG",
                           ignore_dist_thres = 10)
    
    for obj in (target_custom_objs + sampled_distractor_bop_objs):      
        obj.hide(True)