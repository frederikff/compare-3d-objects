# Frederik Frerck 08.10.2022


import os
import csv
import math
import copy
import time
import trimesh
import numpy as np
import pandas as pd
import pyvista as pv
import tkinter as tk
import tkinter.ttk as ttk
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from stl import mesh  # numpy-stl library, not stl library!!
from functools import reduce
from tkinter import filedialog
from tkinter import messagebox
from pyntcloud import PyntCloud
from trimesh import convex
from trimesh import inertia
from trimesh import transformations
from trimesh.exchange import export


class MeshToMatrix:

    @staticmethod
    def generate_line_events(line_list):
        events = []
        for i, line in enumerate(line_list):
            first, second = sorted(line, key=lambda pt: pt[0])
            events.append((first[0], 'start', i))
            events.append((second[0], 'end', i))
        return sorted(events, key=lambda tup: tup[0])

    @staticmethod
    def generate_y(p1, p2, x):
        x1, y1 = p1[:2]
        x2, y2 = p2[:2]
        dy = (y2 - y1)
        dx = (x2 - x1)
        y = dy * (x - x1) / dx + y1
        return y

    def paint_y_axis(self, lines, pixels, x):
        is_black = False
        target_ys = list(map(lambda line: int(self.generate_y(line[0], line[1], x)), lines))
        target_ys.sort()
        # if len(target_ys) % 2:
        #     print('[Warning] The number of lines is odd')
        #     distances = []
        #     for i in range(len(target_ys) - 1):
        #         distances.append(target_ys[i + 1] - target_ys[i])
        #     min_idx = -min((x, -i) for i, x in enumerate(distances))[1]
        #     del target_ys[min_idx]
        yi = 0
        for target_y in target_ys:
            if is_black:
                # Bulk assign all pixels between yi -> target_y
                pixels[yi:target_y, x] = True
            pixels[target_y][x] = True
            is_black = not is_black
            yi = target_y
        # assert is_black is False, 'an error has occurred at x%s' % x

    def lines_to_voxels(self, line_list, pixels):
        current_line_indices = set()
        x = 0
        for (event_x, status, line_ind) in self.generate_line_events(line_list):
            while event_x - x >= 0:
                lines = reduce(lambda acc, cur: acc + [line_list[cur]], current_line_indices, [])
                self.paint_y_axis(lines, pixels, x)
                x += 1
            if status == 'start':
                assert line_ind not in current_line_indices
                current_line_indices.add(line_ind)
            elif status == 'end':
                assert line_ind in current_line_indices
                current_line_indices.remove(line_ind)

    @staticmethod
    def linear_interpolation(p1, p2, distance):
        """
        :param p1: Point 1
        :param p2: Point 2
        :param distance: Between 0 and 1, Lower numbers return points closer to p1.
        :return: A point on the line between p1 and p2
        """
        return p1 * (1 - distance) + p2 * distance

    def where_line_crosses_z(self, p1, p2, z):
        if p1[2] > p2[2]:
            p1, p2 = p2, p1
        # now p1 is below p2 in z
        if p2[2] == p1[2]:
            distance = 0
        else:
            distance = (z - p1[2]) / (p2[2] - p1[2])
        return self.linear_interpolation(p1, p2, distance)

    # main function for the conversion of mesh into matrix
    def triangle_to_intersecting_lines(self, triangle, height, pixels, lines):
        assert (len(triangle) == 3)
        above = list(filter(lambda pt: pt[2] > height, triangle))
        below = list(filter(lambda pt: pt[2] < height, triangle))
        same = list(filter(lambda pt: pt[2] == height, triangle))
        if len(same) == 3:
            for i in range(0, len(same) - 1):
                for j in range(i + 1, len(same)):
                    lines.append((same[i], same[j]))
        elif len(same) == 2:
            lines.append((same[0], same[1]))
        elif len(same) == 1:
            if above and below:
                side1 = self.where_line_crosses_z(above[0], below[0], height)
                lines.append((side1, same[0]))
            else:
                x = int(same[0][0])
                y = int(same[0][1])
                pixels[y][x] = True
        else:
            cross_lines = []
            for a in above:
                for b in below:
                    cross_lines.append((b, a))
            side1 = self.where_line_crosses_z(cross_lines[0][0], cross_lines[0][1], height)
            side2 = self.where_line_crosses_z(cross_lines[1][0], cross_lines[1][1], height)
            lines.append((side1, side2))

    def paint_z_plane(self, a_mesh, height, plane_shape):
        pixels = np.zeros(plane_shape, dtype=bool)
        lines = []
        for triangle in a_mesh:
            self.triangle_to_intersecting_lines(triangle, height, pixels, lines)
        self.lines_to_voxels(lines, pixels)
        return height, pixels

    @staticmethod
    def generate_tri_events(a_mesh):
        # create data structure for plane sweep
        events = []
        for i, tri in enumerate(a_mesh):
            bottom, middle, top = sorted(tri, key=lambda pt: pt[2])
            events.append((bottom[2], 'start', i))
            events.append((top[2], 'end', i))
        return sorted(events, key=lambda tup: tup[0])

    def mesh_to_plane(self, a_mesh, bounding_box):
        # Note: vol should be addressed with vol[z][y][x]
        vol = np.zeros(bounding_box[::-1], dtype=bool)
        current_mesh_indices = set()
        z = 0
        for event_z, status, tri_ind in self.generate_tri_events(a_mesh):
            while event_z - z >= 0:
                mesh_subset = [a_mesh[ind] for ind in current_mesh_indices]
                _, pixels = self.paint_z_plane(mesh_subset, z, vol.shape[1:])
                vol[z] = pixels
                z += 1
            if status == 'start':
                assert tri_ind not in current_mesh_indices
                current_mesh_indices.add(tri_ind)
            elif status == 'end':
                assert tri_ind in current_mesh_indices
                current_mesh_indices.remove(tri_ind)
        return vol

    def mesh_to_matrix(self, meshes, shape):
        resulting_matrices = []
        for mesh_ind, org_mesh in enumerate(meshes):
            cur_vol = self.mesh_to_plane(org_mesh, shape)
            cur_vol_int = np.multiply(cur_vol, 1, dtype=np.uint8)
            # cur_vol_int = cur_vol.astype(uint8)
            resulting_matrices.append(cur_vol_int)
        return resulting_matrices


# convert two csv or stl objects into two trimesh-meshes
# input: list of two objects as csv or stl file and list of file-formats as string
def make_mesh(input_file_paths, file_form):
    initial_meshes = []
    incorrect_mesh = 0
    for i, input_file_path in enumerate(input_file_paths):
        if file_form[i] == '.csv':
            # .csv to .obj
            cloud = pd.read_csv(input_file_path, skiprows=1, header=None)
            cloud.columns = ["x", "y", "z"]
            cloud = PyntCloud(cloud)
            # first convex hull:
            convex_hull_id = cloud.add_structure("convex_hull")
            convex_hull = cloud.structures[convex_hull_id]
            cloud.mesh = convex_hull.get_mesh()
            cloud.to_file("output.obj", also_save=["mesh"])
            # .obj to .stl
            loaded_mesh = trimesh.load("output.obj", file_type='obj')
            # second convex hull:
            a_mesh = convex.convex_hull(loaded_mesh)
            # save both meshes in a list:
            initial_meshes.append(a_mesh)
            os.remove('output.obj')
        elif file_form[i] == '.stl':
            # a_mesh = trimesh.load_mesh(input_file_path)
            a_mesh = trimesh.load(input_file_path)
            initial_meshes.append(a_mesh)
            # if a_mesh.is_watertight:
            #     initial_meshes.append(a_mesh)
            # else:
            #     a_mesh = convex.convex_hull(a_mesh)
            #     initial_meshes.append(a_mesh)
            #     incorrect_mesh += 1

    return initial_meshes, incorrect_mesh


# calculate x, y and z lengths of the bounding boxes of two trimesh-meshes
def calculate_shape(meshes):
    mesh_min = meshes[0].min(axis=(0, 1))
    mesh_max = meshes[0].max(axis=(0, 1))
    for a_mesh in meshes[1:]:
        mesh_min = np.minimum(mesh_min, a_mesh.min(axis=(0, 1)))
        mesh_max = np.maximum(mesh_max, a_mesh.max(axis=(0, 1)))

    bounding_box = mesh_max - mesh_min
    bounding_box *= 2  # 1 would be boundary box
    shape = np.array(bounding_box)
    for i, x in np.ndenumerate(shape):
        shape[i] = math.ceil(x)
    shape = np.floor(shape).astype(int)

    return shape


# calculate the main inertia axes for two trimesh-meshes related to the lowest moment of inertia
def calculate_rotation(meshes):
    inertia_matrix_0 = meshes[0].moment_inertia
    inertia_matrix_1 = meshes[1].moment_inertia
    moments_0, axes_0 = inertia.principal_axis(inertia_matrix_0)
    moments_1, axes_1 = inertia.principal_axis(inertia_matrix_1)
    sort_0 = sorted(zip(moments_0, axes_0), key=lambda x: x[0])
    sorted_moments_0, sorted_axes_0 = zip(*sort_0)
    sort_1 = sorted(zip(moments_1, axes_1), key=lambda y: y[0])
    sorted_moments_1, sorted_axes_1 = zip(*sort_1)

    return sorted_axes_0, sorted_axes_1


# rotate a trimesh-mesh in the same way vector v1 would need to be rotated to point in the direction of vector v2
def rotate(the_mesh, v1, v2):
    rotation_matrix = transformations.rotation_matrix(
        transformations.angle_between_vectors(v1, v2),
        transformations.vector_product(v1, v2))
    the_mesh.apply_transform(rotation_matrix)


# scale both meshes so that the volumes of them equal the target volume parameter
# input: list of two meshes, where the first mesh is the larger one (volume-wise); target-volume
def scale(initial_meshes, target_volume):
    for i in range(2):
        vol_i = initial_meshes[i].volume
        if target_volume > vol_i:
            percentage = vol_i / target_volume
            while round(percentage, 5) < 1.0:
                factor = 1 + (-math.log(percentage) / 10)
                initial_meshes[i].apply_scale(factor)
                # compute volume of the smaller object:
                vol_i = initial_meshes[i].volume
                # new volume-similarity for loop:
                percentage = vol_i / target_volume
        elif target_volume < vol_i:
            percentage = target_volume / vol_i
            while round(percentage, 5) < 1.0:
                factor = 1 - (-math.log(percentage) / 10)
                if factor <= 0.00001:
                    factor = 0.01
                initial_meshes[i].apply_scale(factor)
                # compute volume of the smaller object:
                vol_i = initial_meshes[i].volume
                # new volume-similarity for loop:
                percentage = target_volume / vol_i


# move both meshes so that their center of mass equals the origin
def move(initial_meshes):
    # compute the center of mass
    center_of_mass_1 = initial_meshes[0].center_mass
    center_of_mass_2 = initial_meshes[1].center_mass
    # shift meshes to the same center of mass:
    initial_meshes[0].apply_translation(-center_of_mass_1)
    initial_meshes[1].apply_translation(-center_of_mass_2)


# compute the volume-similarity:
def compute_vol_sim(initial_meshes):
    vol_1 = initial_meshes[0].volume
    vol_2 = initial_meshes[1].volume
    larger = 0
    if vol_1 < vol_2:
        vol_similarity = (vol_1 / vol_2) * 100
        larger = 2
    elif vol_1 > vol_2:
        vol_similarity = (vol_2 / vol_1) * 100
        larger = 1
    else:
        vol_similarity = 100

    return vol_similarity, larger


# calculate how two trimesh-meshes need to be shifted,
# so that they are fully located in the positive area of cartesian coordinates
def offset(meshes):
    one = meshes[0].extents
    two = meshes[1].extents
    x = max(one[0], two[0])
    y = max(one[1], two[1])
    z = max(one[2], two[2])
    resulting_offset = [x, y, z]
    return resulting_offset


# compute a matrix of 0s, 1s, 2s, and 3s out of two matrices of 0s and 1s
def get_comparison_matrix(matrices):
    # matrices contain the finally shifted and scaled objects
    matrix2 = np.where(matrices[1] > 0, 2, matrices[1])  # matrix of 0s and 2s
    m_with_distinction = matrices[0] + matrix2  # matrices[0] is of 0s and 1s
    # in m_with_more_distinction: 1s and 2s is where the objects differ, 3s is where they match/overlap
    return m_with_distinction


# calculate the shape-similarity of two objects as matrices (containing 1s for material and 0s for no material)
# input: list of matrices as lists of lists of lists (3D)
def shape_comparison(matrices):
    # matrices contain the finally shifted and scaled objects
    m = matrices[0] + matrices[1]  # 1s is where the objects differ, 2s is where they match/overlap

    flat_array_1 = matrices[0].flatten()
    vol_1_list = flat_array_1.tolist()
    count_1s_in_1 = vol_1_list.count(1)
    flat_array_2 = matrices[1].flatten()
    vol_2_list = flat_array_2.tolist()
    count_1s_in_2 = vol_2_list.count(1)
    reference = (count_1s_in_1 + count_1s_in_2) / 2
    flat_array = m.flatten()
    vol_list = flat_array.tolist()
    count_2s = vol_list.count(2)
    sim = (count_2s / reference) * 100

    return sim


# convert trimesh-mesh to stl-mesh so that it can be converted to a matrix:
# required by the MeshToMatrix class
def trimesh_to_stl_mesh(the_meshes):
    export.export_mesh(the_meshes[0], 'output_1.stl', file_type=None)
    export.export_mesh(the_meshes[1], 'output_2.stl', file_type=None)
    new_mesh_1 = mesh.Mesh.from_file('output_1.stl')
    new_mesh_2 = mesh.Mesh.from_file('output_2.stl')
    os.remove('output_1.stl')
    os.remove('output_2.stl')

    converted_mesh_1 = np.hstack((new_mesh_1.v0[:, np.newaxis],
                                  new_mesh_1.v1[:, np.newaxis],
                                  new_mesh_1.v2[:, np.newaxis]))
    converted_mesh_2 = np.hstack((new_mesh_2.v0[:, np.newaxis],
                                  new_mesh_2.v1[:, np.newaxis],
                                  new_mesh_2.v2[:, np.newaxis]))
    the_new_meshes = [converted_mesh_1, converted_mesh_2]

    return the_new_meshes


# compute the shape-similarity for each possible orientation of the bounding-box:
def bounding_box_comparison(bb_meshes, degs, turn_back):
    # rotate bounding box and object:
    qx = transformations.quaternion_about_axis(degs[0], [1, 0, 0])
    qy = transformations.quaternion_about_axis(degs[1], [0, 1, 0])
    qz = transformations.quaternion_about_axis(degs[2], [0, 0, 1])
    q = transformations.quaternion_multiply(qx, qy)
    q = transformations.quaternion_multiply(q, qz)
    r = transformations.quaternion_matrix(q)
    bb_meshes[1].apply_transform(r)

    com_1 = bb_meshes[0].center_mass
    com_2 = bb_meshes[1].center_mass
    bb_meshes[0].apply_translation(-com_1)
    bb_meshes[1].apply_translation(-com_2)
    extends_1 = bb_meshes[0].extents
    extends_2 = bb_meshes[1].extents
    o_x = max(extends_1[0], extends_2[0])
    o_y = max(extends_1[1], extends_2[1])
    o_z = max(extends_1[2], extends_2[2])
    bb_offset = [o_x, o_y, o_z]
    bb_meshes[0].apply_translation(bb_offset)
    bb_meshes[1].apply_translation(bb_offset)

    # calculate shape similarity and compare:
    stl_meshes_bb = trimesh_to_stl_mesh(bb_meshes)
    shape_bb = calculate_shape(stl_meshes_bb)
    mtm = MeshToMatrix()
    start = time.time()
    matrices_bb = mtm.mesh_to_matrix(stl_meshes_bb, shape_bb)
    end = time.time()
    mtm_time = end - start
    shape_similarity_bb_new = shape_comparison(matrices_bb)

    if turn_back == 1:
        qx = transformations.quaternion_about_axis(-degs[0], [1, 0, 0])
        qy = transformations.quaternion_about_axis(-degs[1], [0, 1, 0])
        qz = transformations.quaternion_about_axis(-degs[2], [0, 0, 1])
        q = transformations.quaternion_multiply(qz, qy)
        q = transformations.quaternion_multiply(q, qx)
        r = transformations.quaternion_matrix(q)
        bb_meshes[1].apply_transform(r)

    return shape_similarity_bb_new, mtm_time


# execute the normalisation of two trimesh-meshes:
# the shape-comparison os also initialised together with the normalisation
# input: the objects, the target-volume and the bounding-box option
def process_meshes(the_meshes, target, if_bb):

    move(the_meshes)

    bounding_box_meshes = copy.deepcopy(the_meshes)
    scale(bounding_box_meshes, target)
    shape_similarity_bb = 0

    # calculate the principal axis of inertia:
    inertia_axes_0, inertia_axes_1 = calculate_rotation(the_meshes)

    turned_meshes = copy.deepcopy(the_meshes)

    # rotate meshes so that the first inertia axis is aligned to the z-axis:
    if inertia_axes_0[0][0] != 0 or inertia_axes_0[0][1] != 0:
        rotate(the_meshes[0], inertia_axes_0[0], [0, 0, 1])
        rotate(turned_meshes[0], inertia_axes_0[0], [0, 0, 1])
    # object 2 gets by 180 degrees differently orientated:
    if inertia_axes_1[0][0] != 0 or inertia_axes_1[0][1] != 0:
        rotate(the_meshes[1], inertia_axes_1[0], [0, 0, 1])
        rotate(turned_meshes[1], inertia_axes_1[0], [0, 0, -1])
    elif inertia_axes_1[0][0] == 0 and inertia_axes_1[0][1] == 0:
        rotate(turned_meshes[1], inertia_axes_1[0], [1, 0, 0])
        rotate(turned_meshes[1], inertia_axes_1[0], [0, 0, -1])

    inertia_axes_0, inertia_axes_1 = calculate_rotation(the_meshes)

    # rotation of the first object around the z-axis
    # the new inertia-axes 2 and three now lie in the x-y-plane (z-parameter = 0)
    if inertia_axes_0[1][0] != 0:
        # normalization for unit circle calculations:
        normalized_0_1 = inertia_axes_0[1] / np.linalg.norm(inertia_axes_0[1])
        if normalized_0_1[0] < 0:
            alpha = (math.asin(normalized_0_1[1]) - (math.pi / 2)) / math.pi * 180  # angle calculation
        else:
            alpha = (-1) * (math.asin(normalized_0_1[1]) - (math.pi / 2)) / math.pi * 180
        q = transformations.quaternion_about_axis(np.radians(alpha), inertia_axes_0[0])
        # alternative for inertia_axes_0[0] as rotation axis is the [0, 0, 1] axis itself
        r = transformations.quaternion_matrix(q)  # execution of the rotation with quaternions and transformation-matrix
        the_meshes[0].apply_transform(r)
        turned_meshes[0] = the_meshes[0]

    # rotation of the second object around the z-axis
    inertia_axes_0, inertia_axes_1 = calculate_rotation(the_meshes)
    if inertia_axes_1[1][0] != 0:
        normalized_1_1 = inertia_axes_1[1] / np.linalg.norm(inertia_axes_1[1])
        if normalized_1_1[0] < 0:
            alpha = (math.asin(normalized_1_1[1]) - (math.pi / 2)) / math.pi * 180
        else:
            alpha = (-1) * (math.asin(normalized_1_1[1]) - (math.pi / 2)) / math.pi * 180
        q = transformations.quaternion_about_axis(np.radians(alpha), inertia_axes_1[0])
        # alternative for inertia_axes_1[0] as rotation axis is the [0, 0, 1] axis itself
        r = transformations.quaternion_matrix(q)
        the_meshes[1].apply_transform(r)

    # rotation of the second object around the z-axis (by 180 degree differently rotated version)
    inertia_axes_turned_0, inertia_axes_turned_1 = calculate_rotation(turned_meshes)
    if inertia_axes_turned_1[1][0] != 0:
        normalized_turned_1_1 = inertia_axes_turned_1[1] / np.linalg.norm(inertia_axes_turned_1[1])
        if normalized_turned_1_1[0] < 0:
            alpha = (math.asin(normalized_turned_1_1[1]) - (math.pi / 2)) / math.pi * 180
        else:
            alpha = (-1) * (math.asin(normalized_turned_1_1[1]) - (math.pi / 2)) / math.pi * 180
        q = transformations.quaternion_about_axis(np.radians(alpha), inertia_axes_turned_1[0])
        # alternative for inertia_axes_turned_1[0] as rotation axis is the [0, 0, 1] axis itself
        r = transformations.quaternion_matrix(q)
        turned_meshes[1].apply_transform(r)

    # add the 180 degree options for the second part of the inertia-normalisation
    the_meshes_rotated = copy.deepcopy(the_meshes)
    q = transformations.quaternion_about_axis(np.radians(180), inertia_axes_turned_1[0])
    r = transformations.quaternion_matrix(q)
    the_meshes_rotated[1].apply_transform(r)
    turned_meshes_rotated = copy.deepcopy(turned_meshes)
    q = transformations.quaternion_about_axis(np.radians(180), inertia_axes_turned_1[0])
    r = transformations.quaternion_matrix(q)
    turned_meshes_rotated[1].apply_transform(r)

    scale(the_meshes, target)
    move(the_meshes)
    shift = offset(the_meshes)
    the_meshes[0].apply_translation(shift)
    the_meshes[1].apply_translation(shift)

    scale(turned_meshes, target)
    move(turned_meshes)
    shift = offset(turned_meshes)
    turned_meshes[0].apply_translation(shift)
    turned_meshes[1].apply_translation(shift)

    scale(the_meshes_rotated, target)
    move(the_meshes_rotated)
    shift = offset(the_meshes_rotated)
    the_meshes_rotated[0].apply_translation(shift)
    the_meshes_rotated[1].apply_translation(shift)

    scale(turned_meshes_rotated, target)
    move(turned_meshes_rotated)
    shift = offset(turned_meshes_rotated)
    turned_meshes_rotated[0].apply_translation(shift)
    turned_meshes_rotated[1].apply_translation(shift)

    # bounding box normalisation:
    count_mtm_time = 0
    if if_bb == 1:
        for two_times in range(2):
            transform_1, extends_1 = trimesh.bounds.oriented_bounds(bounding_box_meshes[0])
            transform_2, extends_2 = trimesh.bounds.oriented_bounds(bounding_box_meshes[1])
            bounding_box_meshes[0].apply_transform(-transform_1)
            bounding_box_meshes[1].apply_transform(-transform_2)

        meshes_bb = trimesh_to_stl_mesh(bounding_box_meshes)
        shape_bb = calculate_shape(meshes_bb)
        mtm = MeshToMatrix()
        start = time.time()
        matrices_bb = mtm.mesh_to_matrix(meshes_bb, shape_bb)
        end = time.time()
        shape_similarity_bb_old = shape_comparison(matrices_bb)
        degrees = [0, 0, 0]
        remember_degrees = [0, 0, 0]
        count_mtm_time += (end - start)
        # 24 possibilities:
        for x in range(2):
            degrees[0] = np.radians(90 * x)
            for y in range(3):
                degrees[1] = np.radians(-90 * y)
                for z in range(4):
                    degrees[2] = np.radians(90 * z)
                    shape_similarity_bb_new, mtm_time = bounding_box_comparison(bounding_box_meshes, degrees, 1)
                    count_mtm_time += mtm_time
                    if shape_similarity_bb_new > shape_similarity_bb_old:
                        shape_similarity_bb_old = shape_similarity_bb_new
                        remember_degrees = [degrees[0], degrees[1], degrees[2]]
        shape_similarity_bb, mtm_time = bounding_box_comparison(bounding_box_meshes, remember_degrees, 0)
        count_mtm_time += mtm_time

    # p_m_1 = pv.wrap(the_meshes[0])
    # p_m_2 = pv.wrap(the_meshes[1])
    # p_m_3 = pv.wrap(turned_meshes[0])
    # p_m_4 = pv.wrap(turned_meshes[1])
    # p_m_5 = pv.wrap(the_meshes_rotated[0])
    # p_m_6 = pv.wrap(the_meshes_rotated[1])
    # p_m_7 = pv.wrap(turned_meshes_rotated[0])
    # p_m_8 = pv.wrap(turned_meshes_rotated[1])
    # p_m_9 = pv.wrap(bounding_box_meshes[0])
    # p_m_10 = pv.wrap(bounding_box_meshes[1])
    # pl = pv.Plotter(shape=(2, 3))
    # _ = pl.add_mesh(p_m_1, color='lightskyblue', opacity=0.4)
    # _ = pl.add_mesh(p_m_2, color='darkblue', opacity=0.4)
    # pl.add_title('inertia alignment', font_size=5, color='black')
    # pl.subplot(0, 1)
    # _ = pl.add_mesh(p_m_3, color='lightskyblue', opacity=0.4)
    # _ = pl.add_mesh(p_m_4, color='darkblue', opacity=0.4)
    # pl.add_title('turned inertia', font_size=5, color='black')
    # pl.subplot(1, 0)
    # _ = pl.add_mesh(p_m_5, color='lightskyblue', opacity=0.4)
    # _ = pl.add_mesh(p_m_6, color='darkblue', opacity=0.4)
    # pl.add_title('rotated inertia alignment', font_size=5, color='black')
    # pl.subplot(1, 1)
    # _ = pl.add_mesh(p_m_7, color='lightskyblue', opacity=0.4)
    # _ = pl.add_mesh(p_m_8, color='darkblue', opacity=0.4)
    # pl.add_title('rotated and turned inertia alignment', font_size=5, color='black')
    # pl.subplot(0, 2)
    # _ = pl.add_mesh(p_m_9, color='lightskyblue', opacity=0.4)
    # _ = pl.add_mesh(p_m_10, color='darkblue', opacity=0.4)
    # pl.add_title('bounding box alignment', font_size=5, color='black')
    # pl.set_background('white')
    # pl.show()

    meshes_the = trimesh_to_stl_mesh(the_meshes)
    meshes_turned = trimesh_to_stl_mesh(turned_meshes)
    meshes_rotated = trimesh_to_stl_mesh(the_meshes_rotated)
    meshes_rotated_turned = trimesh_to_stl_mesh(turned_meshes_rotated)
    meshes_bb = trimesh_to_stl_mesh(bounding_box_meshes)

    # calculate shape similarity for both 180-degree cases:
    shape_the = calculate_shape(meshes_the)
    shape_turned = calculate_shape(meshes_turned)
    shape_rotated = calculate_shape(meshes_rotated)
    shape_rotated_turned = calculate_shape(meshes_rotated_turned)
    shape_bb = calculate_shape(meshes_bb)
    mtm = MeshToMatrix()
    start = time.time()
    final_matrices = mtm.mesh_to_matrix(meshes_the, shape_the)
    final_matrices_turned = mtm.mesh_to_matrix(meshes_turned, shape_turned)
    final_matrices_rotated = mtm.mesh_to_matrix(meshes_rotated, shape_rotated)
    final_matrices_rotated_turned = mtm.mesh_to_matrix(meshes_rotated_turned, shape_rotated_turned)
    final_matrices_bb = mtm.mesh_to_matrix(meshes_bb, shape_bb)
    end = time.time()
    count_mtm_time += (end - start)
    shape_similarity = shape_comparison(final_matrices)
    shape_similarity_turned = shape_comparison(final_matrices_turned)
    shape_similarity_rotated = shape_comparison(final_matrices_rotated)
    shape_similarity_rotated_turned = shape_comparison(final_matrices_rotated_turned)
    # print()
    # print(round(shape_similarity, 4))
    # print(round(shape_similarity_turned, 4))
    # print(round(shape_similarity_rotated, 4))
    # print(round(shape_similarity_rotated_turned, 4))
    # print(round(shape_similarity_bb, 4))

    # compare the shape-similarity of the different pose-normalisations:
    track_sim = 1  # 1=shape_similarity or shape_similarity_turned, 2=shape_similarity_bb or shape_similarity_bb_turned
    if shape_similarity_turned > shape_similarity:
        the_meshes = turned_meshes
        final_matrices = final_matrices_turned
        shape_similarity = shape_similarity_turned
    if shape_similarity_rotated > shape_similarity:
        the_meshes = the_meshes_rotated
        final_matrices = final_matrices_rotated
        shape_similarity = shape_similarity_rotated
    if shape_similarity_rotated_turned > shape_similarity:
        the_meshes = turned_meshes_rotated
        final_matrices = final_matrices_rotated_turned
        shape_similarity = shape_similarity_rotated_turned
    if shape_similarity_bb > shape_similarity:
        the_meshes = bounding_box_meshes
        final_matrices = final_matrices_bb
        shape_similarity = shape_similarity_bb
        track_sim = 2
    elif shape_similarity_bb == shape_similarity:
        track_sim = 21
    print()
    if track_sim == 1:
        print('Pose-normalisation with inertia-axes')
    elif track_sim == 2:
        print('Pose-normalisation with bounding-box')
    elif track_sim == 21:
        print('Pose-normalisations with inertia-axes and with bounding-box equally good')

    return shape_similarity, final_matrices, the_meshes, count_mtm_time


# plot a similarity-matrix containing 0s, 1s, 2s and 3s with plotly
def plot_diff(m, density, transparency, title):
    x_light = math.ceil(m.shape[0] / 2)
    y_light = math.ceil(m.shape[1] / 2)
    z_light = math.ceil(m.shape[2] / 2)
    light_matrix = np.zeros((x_light, y_light, z_light))
    for index, val in np.ndenumerate(m):
        if index[0] % density == 0 and index[1] % density == 0 and index[2] % density == 0:
            if index[0] == 0:
                index_x = 0
            else:
                index_x = int(index[0] / 2)
            if index[1] == 0:
                index_y = 0
            else:
                index_y = int(index[1] / 2)
            if index[2] == 0:
                index_z = 0
            else:
                index_z = int(index[2] / 2)
            light_matrix[index_x][index_y][index_z] = val

    x_max = light_matrix.shape[0]
    y_max = light_matrix.shape[1]
    z_max = light_matrix.shape[2]

    a, b, c = np.mgrid[0:x_max, 0:y_max, 0:z_max]
    values = light_matrix

    if transparency == 0.0:
        t = 0.1
    else:
        t = transparency

    fig = go.Figure(data=go.Volume(
                                   x=a.flatten(),
                                   y=b.flatten(),
                                   z=c.flatten(),
                                   value=values.flatten(),
                                   isomin=1,
                                   isomax=3,
                                   opacity=t,  # needs to be small to see through all surfaces,
                                               # like transparency for plot_differences
                                   surface_count=10,  # needs to be a large number for good volume rendering
                                   ),
                    layout=go.Layout(title=go.layout.Title(text=title))
                    )

    fig.update_layout(scene=dict(
                                 xaxis=dict(
                                            showgrid=False,
                                            showbackground=False,
                                            visible=False,
                                            zeroline=False),
                                 yaxis=dict(
                                            showgrid=False,
                                            showbackground=False,
                                            visible=False,
                                            zeroline=False),
                                 zaxis=dict(
                                            showgrid=False,
                                            showbackground=False,
                                            visible=False,
                                            zeroline=False)
                                 )
                      )

    return fig


# plot a similarity-matrix containing 0s, 1s, 2s and 3s with matplotlib
def plot_differences(m, dense, transparency, name, sim):
    x_one = []
    y_one = []
    z_one = []
    x_two = []
    y_two = []
    z_two = []
    x_three = []
    y_three = []
    z_three = []
    one = 0
    two = 0
    for index, val in np.ndenumerate(m):
        if index[0] % dense == 0 and index[1] % dense == 0 and index[2] % dense == 0:
            if val == 1:
                one = 1
                x_one.append(index[0])
                y_one.append(index[1])
                z_one.append(index[2])
            elif val == 2:
                two = 1
                x_two.append(index[0])
                y_two.append(index[1])
                z_two.append(index[2])
            elif val == 3:
                x_three.append(index[0])
                y_three.append(index[1])
                z_three.append(index[2])

    if one == 0 and two == 0:
        x_max = max(x_three)
        y_max = max(y_three)
        z_max = max(z_three)
        x_min = min(x_three)
        y_min = min(y_three)
        z_min = min(z_three)
    elif one == 0:
        if max(x_two) >= max(x_three):
            x_max = max(x_two)
        else:
            x_max = max(x_three)
        if min(x_two) <= min(x_three):
            x_min = min(x_two)
        else:
            x_min = min(x_three)
        if max(y_two) >= max(y_three):
            y_max = max(y_two)
        else:
            y_max = max(y_three)
        if min(y_two) <= min(y_three):
            y_min = min(y_two)
        else:
            y_min = min(y_three)
        if max(z_two) >= max(z_three):
            z_max = max(z_two)
        else:
            z_max = max(z_three)
        if min(z_two) <= min(z_three):
            z_min = min(z_two)
        else:
            z_min = min(z_three)
    elif two == 0:
        if max(x_one) >= max(x_three):
            x_max = max(x_one)
        else:
            x_max = max(x_three)
        if min(x_one) <= min(x_three):
            x_min = min(x_one)
        else:
            x_min = min(x_three)
        if max(y_one) >= max(y_three):
            y_max = max(y_one)
        else:
            y_max = max(y_three)
        if min(y_one) <= min(y_three):
            y_min = min(y_one)
        else:
            y_min = min(y_three)
        if max(z_one) >= max(z_three):
            z_max = max(z_one)
        else:
            z_max = max(z_three)
        if min(z_one) <= min(z_three):
            z_min = min(z_one)
        else:
            z_min = min(z_three)
    else:
        if max(x_one) >= max(x_two) and max(x_one) >= max(x_three):
            x_max = max(x_one)
        elif max(x_two) >= max(x_three):
            x_max = max(x_two)
        else:
            x_max = max(x_three)
        if min(x_one) <= min(x_two) and min(x_one) <= min(x_three):
            x_min = min(x_one)
        elif min(x_two) <= min(x_three):
            x_min = min(x_two)
        else:
            x_min = min(x_three)
        if max(y_one) >= max(y_two) and max(y_one) >= max(y_three):
            y_max = max(y_one)
        elif max(y_two) >= max(y_three):
            y_max = max(y_two)
        else:
            y_max = max(y_three)
        if min(y_one) <= min(y_two) and min(y_one) <= min(y_three):
            y_min = min(y_one)
        elif min(y_two) <= min(y_three):
            y_min = min(y_two)
        else:
            y_min = min(y_three)
        if max(z_one) >= max(z_two) and max(z_one) >= max(z_three):
            z_max = max(z_one)
        elif max(z_two) >= max(z_three):
            z_max = max(z_two)
        else:
            z_max = max(z_three)
        if min(z_one) <= min(z_two) and min(z_one) <= min(z_three):
            z_min = min(z_one)
        elif min(z_two) <= min(z_three):
            z_min = min(z_two)
        else:
            z_min = min(z_three)

    x_scale = [x_min, x_max]
    y_scale = [y_min, y_max]
    z_scale = [z_min, z_max]

    fig = plt.figure()
    manager = plt.get_current_fig_manager()
    manager.window.state('zoomed')
    ax = fig.add_subplot(111, projection='3d', title=name)
    ax.set_box_aspect((np.ptp(x_scale), np.ptp(y_scale), np.ptp(z_scale)))  # aspect ratio is 1:1:1 in data space
    plt.subplots_adjust(right=0.777, left=0, bottom=0.1, top=0.877, wspace=0.2, hspace=0.2)
    if transparency == 0.0:
        if dense == 1:
            t = 0.01
        elif dense == 2:
            t = 0.07
        elif dense == 3:
            t = 0.14
        else:
            t = 0.21
    else:
        t = transparency
    if sim <= 70:  # 0-70
        ax.scatter(x_one, y_one, z_one, c='skyblue', alpha=t * 2, marker='o')  # alternative: marker='.'
        ax.scatter(x_two, y_two, z_two, c='royalblue', alpha=t * 2, marker='o')
        ax.scatter(x_three, y_three, z_three, c='black', alpha=1, marker='o')
    elif sim <= 80:  # 70-80
        ax.scatter(x_one, y_one, z_one, c='skyblue', alpha=t * 3, marker='o')
        ax.scatter(x_two, y_two, z_two, c='royalblue', alpha=t * 3, marker='o')
        ax.scatter(x_three, y_three, z_three, c='black', alpha=0.3 - (3 / dense * 0.1) + t * 2, marker='o')
    elif sim <= 90:  # 80-90
        ax.scatter(x_one, y_one, z_one, c='deepskyblue', alpha=t * dense, marker='o')
        ax.scatter(x_two, y_two, z_two, c='mediumblue', alpha=t * dense, marker='o')
        ax.scatter(x_three, y_three, z_three, c='dimgray', alpha=t * (dense / 2), marker='o')
    else:  # 95-100
        ax.scatter(x_one, y_one, z_one, c='deepskyblue', alpha=0.5 - (3 / dense * 0.1) + t * 2, marker='o')
        ax.scatter(x_two, y_two, z_two, c='mediumblue', alpha=0.5 - (3 / dense * 0.1) + t * 2, marker='o')
        ax.scatter(x_three, y_three, z_three, c='dimgray', alpha=t * 1.5, marker='o')

    plt.axis('off')


# organise the output and the illustration with the different plotting options
def illustrate(m_wmd, final_meshes, original_meshes, show, density, transparency, sim_s,
               export_name_1, export_name_2):
    plot_time = 0
    if show == 'matplotlib' or show == 'matplotlib(+export png)':
        title = 'Representation of the two aligned objects, scaled to the same volume'
        start = time.time()
        plot_differences(m_wmd, density, transparency, title, sim_s)  # plotly figure,scatter; comparison with scaling
        if show == 'matplotlib(+export png)':
            plt.savefig(str(export_name_1[0]) + '-vs.-' + str(export_name_2[0])
                        + ' (matplotlib_illustration)' + '.png')
        end = time.time()
        plot_time = end - start
        plt.show()
    elif show == 'plotly(+export)' or show == 'plotly':
        title = 'Representation of the two aligned objects, scaled to the same volume'
        start = time.time()
        figure = plot_diff(m_wmd, density, transparency, title)  # plotly go.Figure
        if show == 'plotly':
            figure.show()
        elif show == 'plotly(+export)':
            figure.write_html(str(export_name_1[0]) + '-vs.-' + str(export_name_2[0])
                              + ' (plotly_illustration)' + '.html', auto_open=True)
        end = time.time()
        plot_time = end - start
    elif show == 'pyvista plotter' or show == '(pyvista plotter(+export png)':  # plot with the pyvista Plotter
        start = time.time()
        p_m_1 = pv.wrap(original_meshes[0])
        p_m_2 = pv.wrap(original_meshes[1])
        p_m_3 = pv.wrap(final_meshes[0])
        p_m_4 = pv.wrap(final_meshes[1])
        pl = pv.Plotter(shape=(1, 2), window_size=[1200, 600])
        _ = pl.add_mesh(p_m_1, color='slateblue', opacity=0.5)
        _ = pl.add_mesh(p_m_2, color='skyblue', opacity=0.5)
        pl.add_title('Representation of the original objects, moved to the same center of mass',
                     font_size=6, color='black')
        pl.subplot(0, 1)
        _ = pl.add_mesh(p_m_3, color='slateblue', opacity=0.5)
        _ = pl.add_mesh(p_m_4, color='skyblue', opacity=0.7)
        pl.add_title('Representation of the two normalised objects, scaled to the same volume',
                     font_size=6, color='black')
        pl.set_background('white')
        pl.camera.zoom(0.9)
        end = time.time()
        plot_time = end - start
        if show == 'pyvista plotter(+export png)':
            pl.show(screenshot=str(export_name_1[0]) + '-vs.-' + str(export_name_2[0]) + ' (pyvista-illustration).png')
        else:
            pl.show()

    return plot_time


# export a shape-similarity matrix
# input: matrix of 0s, 1s, 2s, 3s and the name to save it with
def export_matrix(the_matrix, name):
    with open(name + '.csv', 'w', newline='') as csvfile:
        write = csv.writer(csvfile)
        for z, first_dimension in enumerate(the_matrix):
            for y, second_dimension in enumerate(first_dimension):
                for x, final_dimension in enumerate(second_dimension):
                    if final_dimension != 0:
                        write.writerow([x, y, z, final_dimension])


def main():
    window = tk.Tk()
    window.title("Comparison of 3D objects")
    window.geometry("800x600")
    window.configure(bg='white')
    window.resizable(False, False)
    ws = window.winfo_screenwidth()
    hs = window.winfo_screenheight()
    x = (ws / 2) - 400
    y = (hs / 2) - 330
    window.geometry('+%d+%d' % (x, y))

    frame_00 = tk.Frame(window, bg='white', height=60, width=250)  # choose first file button
    frame_01 = tk.Frame(window, bg='white', height=60, width=20)  # fill
    frame_02 = tk.Frame(window, bg='white', height=60, width=250)  # choose second file button
    frame_04 = tk.Frame(window, bg='white', height=60, width=80)  # fill
    frame_05 = tk.Frame(window, bg='white', height=30)  # choose illustration text
    frame_15 = tk.Frame(window, bg='white', height=30)  # choose illustration
    frame_20 = tk.Frame(window, bg='white', height=60, width=250)  # first chosen file
    frame_21 = tk.Frame(window, bg='white', height=60, width=20)  # fill
    frame_22 = tk.Frame(window, bg='white', height=60, width=250)  # second chosen file
    frame_24 = tk.Frame(window, bg='white', height=60, width=80)  # fill
    frame_35 = tk.Frame(window, bg='white', height=10)  # extra settings text
    frame_45 = tk.Frame(window, bg='white', height=30)  # density text
    frame_46 = tk.Frame(window, bg='white', height=30)  # set density
    frame_50 = tk.Frame(window, bg='white', height=30)
    frame_52 = tk.Frame(window, bg='white', height=30, width=250)  # Set target-volume text
    frame_55 = tk.Frame(window, bg='white', height=30)  # transparency text
    frame_56 = tk.Frame(window, bg='white', height=30)  # set transparency
    frame_60 = tk.Frame(window, bg='white', height=30)  # info box
    frame_62 = tk.Frame(window, bg='white', height=30)  # Set target-volume
    frame_63 = tk.Frame(window, bg='white', height=30)  # volume unit
    frame_70 = tk.Frame(window, bg='white', height=30)  # restrictions box
    frame_75 = tk.Frame(window, bg='white', height=30)  # decide matrix export
    frame_80 = tk.Frame(window, bg='white', height=30)  # settings-info box
    frame_82 = tk.Frame(window, bg='white', height=30)  # decide bounding box
    frame_85 = tk.Frame(window, bg='white', height=30)  # decide results export
    frame_90 = tk.Frame(window, bg='white', height=50)  # fill
    frame_100 = tk.Frame(window, bg='white', height=60)  # Start comparison button
    frame_110 = tk.Frame(window, bg='white', height=30)  # fill
    frame_120 = tk.Frame(window, bg='white', height=40)  # results header
    frame_130 = tk.Frame(window, bg='white', height=30, width=200)  # larger volume text
    frame_133 = tk.Frame(window, bg='white', height=30, width=125)  # over all computation time text
    frame_135 = tk.Frame(window, bg='white', height=30, width=100)  # over all computation time
    frame_140 = tk.Frame(window, bg='white', height=30, width=100)  # volume-similarity text
    frame_141 = tk.Frame(window, bg='white', height=30, width=100)  # volume-similarity result
    frame_143 = tk.Frame(window, bg='white', height=30, width=125)  # time to mesh_to_matrix text
    frame_145 = tk.Frame(window, bg='white', height=30, width=100)  # time to mesh_to_matrix
    frame_150 = tk.Frame(window, bg='white', height=30, width=100)  # shape-similarity text
    frame_151 = tk.Frame(window, bg='white', height=30, width=100)  # shape-similarity result
    frame_153 = tk.Frame(window, bg='white', height=30, width=125)  # time to plt text
    frame_155 = tk.Frame(window, bg='white', height=30, width=100)  # time to plt
    frame_160 = tk.Frame(window, bg='white', height=30, width=100)  # convex-hull text
    frame_163 = tk.Frame(window, bg='white', height=30, width=125)  # time to export text
    frame_165 = tk.Frame(window, bg='white', height=30, width=100)  # time to export
    frame_170 = tk.Frame(window, bg='white', width=250)  # fill

    frame_00.grid(row=0, column=0, sticky="nsew", rowspan=2)  # choose first file button
    frame_01.grid(row=0, column=1, sticky="nsew", rowspan=2)  # fill
    frame_02.grid(row=0, column=2, sticky="nsew", rowspan=2, columnspan=2)  # choose second file button
    frame_04.grid(row=0, column=4, sticky="nsew", rowspan=2)  # fill
    frame_05.grid(row=0, column=5, sticky="nsew", columnspan=2)  # choose illustration text
    frame_15.grid(row=1, column=5, sticky="nsew", columnspan=2)  # choose illustration
    frame_20.grid(row=2, column=0, sticky="nsew", rowspan=2)  # first chosen file
    frame_21.grid(row=2, column=1, sticky="nsew", rowspan=2)  # fill
    frame_22.grid(row=2, column=2, sticky="nsew", rowspan=2, columnspan=2)  # second chosen file
    frame_24.grid(row=2, column=4, sticky="nsew", rowspan=2)  # fill
    frame_35.grid(row=3, column=5, sticky="nsew", columnspan=2)  # extra settings text
    frame_45.grid(row=4, column=5, sticky="nsew")  # density text
    frame_46.grid(row=4, column=6, sticky="nsew")  # set density
    frame_50.grid(row=5, column=0, sticky="nsew")
    frame_52.grid(row=5, column=2, sticky="nsew", columnspan=2)  # Set target-volume text
    frame_55.grid(row=5, column=5, sticky="nsew")  # transparency text
    frame_56.grid(row=5, column=6, sticky="nsew")  # set transparency
    frame_60.grid(row=6, column=0, sticky="nsew")  # info box
    frame_62.grid(row=6, column=2, sticky="nsew")  # Set target-volume
    frame_63.grid(row=6, column=3, sticky="nsew")  # volume unit
    frame_70.grid(row=7, column=0, sticky="nsew")  # restrictions box
    frame_75.grid(row=7, column=5, sticky="nsew", columnspan=2)  # decide matrix export
    frame_80.grid(row=8, column=0, sticky="nsew")  # settings-info box
    frame_82.grid(row=8, column=2, sticky="nsew")  # decide  bounding box
    frame_85.grid(row=8, column=5, sticky="nsew", columnspan=2)  # decide results export
    frame_90.grid(row=9, column=0, sticky="nsew")  # fill
    frame_100.grid(row=10, column=0, sticky="nsew", columnspan=7)  # Start comparison button
    frame_110.grid(row=11, column=0, sticky="nsew")  # fill
    frame_120.grid(row=12, column=0, sticky="nsew", columnspan=6)  # results header
    frame_130.grid(row=13, column=0, sticky="nsew",  columnspan=2)  # larger volume text
    frame_133.grid(row=13, column=3, sticky="nsew", columnspan=2)  # over all computation time text
    frame_135.grid(row=13, column=5, sticky="nsew")  # over all computation time
    frame_140.grid(row=14, column=0, sticky="nsew")  # volume-similarity text
    frame_141.grid(row=14, column=1, sticky="nsew", columnspan=2)  # volume-similarity result
    frame_143.grid(row=14, column=3, sticky="nsew", columnspan=2)  # time to mesh_to_matrix text
    frame_145.grid(row=14, column=5, sticky="nsew")  # time to mesh_to_matrix
    frame_150.grid(row=15, column=0, sticky="nsew")  # shape-similarity text
    frame_151.grid(row=15, column=1, sticky="nsew", columnspan=2)  # shape-similarity result
    frame_153.grid(row=15, column=3, sticky="nsew", columnspan=2)  # time to plot text
    frame_155.grid(row=15, column=5, sticky="nsew")  # time to plot
    frame_160.grid(row=16, column=0, sticky="nsew")  # shape-similarity text
    frame_163.grid(row=16, column=3, sticky="nsew", columnspan=2)  # time to export text
    frame_165.grid(row=16, column=5, sticky="nsew")  # time to export
    frame_170.grid(row=17, column=0, sticky="nsew")  # fill

    frame_20.config(highlightbackground='grey', highlightthickness=3)
    frame_22.config(highlightbackground='grey', highlightthickness=3)

    frame_20.pack_propagate(False)
    name_1 = tk.Label(frame_20, bg='white')
    name_1.pack(fill='both', expand=True)
    frame_22.pack_propagate(False)
    name_2 = tk.Label(frame_22, bg='white')
    name_2.pack(fill='both', expand=True)

    frame_05.pack_propagate(False)
    choose_illustration_text = tk.Label(frame_05, bg='white')
    choose_illustration_text.pack(side='left', fill='both', expand=False)
    choose_illustration_text.configure(text='Choose the illustration-method:', font=('Aerial', 10))
    frame_15.pack_propagate(False)
    style = ttk.Style()
    style.theme_use('alt')
    style.configure('TCombobox', background='lightgray', fieldbackground='lightgray', selectbackground='none',
                    foreground='black', fieldforeground='black', selectforeground='black')
    selected_illustration = tk.StringVar()
    choose_illustration = ttk.Combobox(frame_15, textvariable=selected_illustration, state='readonly')
    choose_illustration['values'] = ('pyvista plotter', 'matplotlib', 'plotly', 'plotly(+export)',
                                     'none', 'matplotlib(+export png)', 'pyvista plotter(+export png)')
    choose_illustration.current(0)  # set the selected item
    choose_illustration.pack(fill='both', expand=True)

    frame_35.pack_propagate(False)
    density_text = tk.Label(frame_35, bg='white')
    density_text.pack(side='left', fill='both', expand=False)
    density_text.configure(text='Settings for matplotlib and plotly:', font=('Aerial', 10))

    # density of points in 3D plot, 1 displays all points
    frame_45.pack_propagate(False)
    density_text = tk.Label(frame_45, bg='white')
    density_text.pack(side='left', fill='both', expand=False)
    density_text.configure(text='Density:', font=('Aerial', 10))
    frame_46.pack_propagate(False)
    var_1 = tk.IntVar()
    var_1.set(2)
    get_density = tk.Spinbox(frame_46, from_=1, to=5, increment=1, width=3, textvariable=var_1, state='readonly')
    get_density.pack(side='right', fill='x', expand=True)
    get_density.configure(font=('Aerial', 10))
    # density = float(var_1.get())

    # transparency of differing points (has to be between 0.01 and 1)
    frame_55.pack_propagate(False)
    transparency_text = tk.Label(frame_55, bg='white')
    transparency_text.pack(side='left', fill='both', expand=False)
    transparency_text.configure(text='Transparency:', font=('Aerial', 10))
    frame_56.pack_propagate(False)
    var_2 = tk.StringVar()
    var_2.set('0.0')
    get_transparency = tk.Spinbox(frame_56, from_=0.0, to=0.5, width=3, textvariable=var_2,
                                  format='%.2f', increment=0.01, state='readonly')
    get_transparency.pack(side='right', fill='x', expand=True)
    get_transparency.configure(font=('Aerial', 10))
    # transparency = float(var_2.get())

    frame_60.pack_propagate(False)
    info_box = tk.Label(frame_60, bg='white')
    info_box.pack(side='left', fill='both', expand=False)

    frame_52.pack_propagate(False)
    density_text = tk.Label(frame_52, bg='white')
    density_text.pack(side='left', fill='both', expand=False)
    density_text.configure(text='Set scaling-target-volume:', font=('Aerial', 10))

    frame_70.pack_propagate(False)
    info_box = tk.Label(frame_70, bg='white')
    info_box.pack(side='left', fill='both', expand=False)

    frame_62.pack_propagate(False)
    tar_var = tk.IntVar()
    tar_var.set(100000)
    get_target = tk.Entry(frame_62, width=3, textvariable=tar_var)
    get_target.pack(side='right', fill='x', expand=True)
    get_target.configure(font=('Aerial', 10), bg='gray85')

    frame_63.pack_propagate(False)
    density_text = tk.Label(frame_63, bg='white')
    density_text.pack(side='left', fill='both', expand=False)
    density_text.configure(text='mm^3', font=('Aerial', 9))

    frame_75.pack_propagate(False)
    check_decision = tk.IntVar()
    export_decision = tk.Checkbutton(frame_75, text='Export comparison matrices', variable=check_decision,
                                     bg='white', activebackground='white')
    export_decision.pack(side='left', fill='both', expand=False)
    export_decision.configure(font=('Aerial', 10))

    frame_80.pack_propagate(False)
    info_box = tk.Label(frame_80, bg='white')
    info_box.pack(side='left', fill='both', expand=False)

    frame_82.pack_propagate(False)
    check_check = tk.IntVar()
    check_check.set(1)
    bb_decision = tk.Checkbutton(frame_82, text='Use bounding box', variable=check_check,
                                 bg='white', activebackground='white', onvalue=1, offvalue=0)
    bb_decision.pack(side='left', fill='both', expand=False)
    bb_decision.configure(font=('Aerial', 10))

    frame_85.pack_propagate(False)
    check_if = tk.IntVar()
    results_export_decision = tk.Checkbutton(frame_85, text='Export results', variable=check_if,
                                             bg='white', activebackground='white')
    results_export_decision.pack(side='left', fill='both', expand=False)
    results_export_decision.configure(font=('Aerial', 10))

    frame_120.pack_propagate(False)
    names = tk.Label(frame_120, bg='white')
    names.pack(side='left', fill='both', expand=False)
    names.configure(font=('Aerial', 11, 'underline'))

    frame_130.pack_propagate(False)
    larger_text = tk.Label(frame_130, bg='white')
    larger_text.pack(side='left', fill='both', expand=False)
    larger_text.configure(font=('Aerial', 10))

    frame_140.pack_propagate(False)
    similarity_volume_text = tk.Label(frame_140, bg='white')
    similarity_volume_text.pack(side='left', fill='both', expand=False)
    similarity_volume_text.configure(text='Volume-similarity:', font=('Aerial', 12))
    frame_150.pack_propagate(False)
    similarity_shape_text = tk.Label(frame_150, bg='white')
    similarity_shape_text.pack(side='left', fill='both', expand=False)
    similarity_shape_text.configure(text='Shape-similarity:', font=('Aerial', 12))

    frame_160.pack_propagate(False)
    convex_text = tk.Label(frame_160, bg='white')
    convex_text.pack(side='left', fill='both', expand=False)
    convex_text.configure(font=('Aerial', 10))

    frame_133.pack_propagate(False)
    computation_time_text = tk.Label(frame_133, bg='white')
    computation_time_text.pack(side='left', fill='both', expand=False)
    computation_time_text.configure(text='Computation time:', font=('Aerial', 10))
    frame_143.pack_propagate(False)
    matrix_time_text = tk.Label(frame_143, bg='white')
    matrix_time_text.pack(side='left', fill='both', expand=False)
    matrix_time_text.configure(text='Time to "mesh-to-matrix":', font=('Aerial', 10))
    frame_153.pack_propagate(False)
    plot_time_text = tk.Label(frame_153, bg='white')
    plot_time_text.pack(side='left', fill='both', expand=False)
    plot_time_text.configure(text='Time to plot and export:', font=('Aerial', 10))
    frame_163.pack_propagate(False)
    export_time_text = tk.Label(frame_163, bg='white')
    export_time_text.pack(side='left', fill='both', expand=False)
    export_time_text.configure(text='Time to export matrix+results:', font=('Aerial', 10))

    frame_141.pack_propagate(False)
    similarity_volume = tk.Label(frame_141, bg='white')
    similarity_volume.pack(side='left', fill='both', expand=False)
    similarity_volume.configure(text='0 %', font=('Aerial', 12))
    frame_151.pack_propagate(False)
    similarity_shape = tk.Label(frame_151, bg='white')
    similarity_shape.pack(side='left', fill='both', expand=False)
    similarity_shape.configure(text='0 %', font=('Aerial', 12))

    frame_135.pack_propagate(False)
    computation_time = tk.Label(frame_135, bg='white')
    computation_time.pack(side='left', fill='both', expand=False)
    computation_time.configure(text='0 sec', font=('Aerial', 10))
    frame_145.pack_propagate(False)
    matrix_time = tk.Label(frame_145, bg='white')
    matrix_time.pack(side='left', fill='both', expand=False)
    matrix_time.configure(text='0 sec', font=('Aerial', 10))
    frame_155.pack_propagate(False)
    plot_time = tk.Label(frame_155, bg='white')
    plot_time.pack(side='left', fill='both', expand=False)
    plot_time.configure(text='0 sec', font=('Aerial', 10))
    frame_165.pack_propagate(False)
    export_time = tk.Label(frame_165, bg='white')
    export_time.pack(side='left', fill='both', expand=False)
    export_time.configure(text='0 sec', font=('Aerial', 10))

    frame_170.pack_propagate(False)
    info_box = tk.Label(frame_170, bg='white')
    info_box.pack(side='left', fill='both', expand=False)

    file_1 = []
    basename_1 = []
    file_2 = []
    basename_2 = []

    def clicked_button_first():
        file = filedialog.askopenfilename(initialdir=os.path.dirname(os.path.realpath("__file__")),
                                          title="Select a File", filetypes=(("stl files", "*.stl*"),
                                                                            ("csv files", "*.csv*")))
        basename = os.path.basename(file)
        name_1.configure(text=basename, font=('Aerial', 10))
        basename_1.clear()
        basename_1.append(basename)
        file_1.clear()
        if file != '':
            file_1.append(file)

    def clicked_button_second():
        file = filedialog.askopenfilename(initialdir=os.path.dirname(os.path.realpath("__file__")),
                                          title="Select a File", filetypes=(("stl files", "*.stl*"),
                                                                            ("csv files", "*.csv*")))
        basename = os.path.basename(file)
        name_2.configure(text=basename, font=('Aerial', 10))
        basename_2.clear()
        basename_2.append(basename)
        file_2.clear()
        if file != '':
            file_2.append(file)

    def clicked_button_third():
        lines = ['- Volume-similarity = volume of the smaller object',
                 '                                     / sum of both volumes',
                 '- Shape-similarity = overlapping volume (after the objects',
                 '                                                                      where normalised)',
                 '                                   / target-volume (scaling-volume of both'
                 '                                                                    objects)']
        messagebox.showinfo('Information', "\n".join(lines))

    def clicked_button_fourth():
        lines = ['- The accuracy increases with the target volume',
                 '- The accuracy decreases for .csv objects because of the',
                 '  convex hull that is needed for conversion',
                 '- The computation-time increases with the size of the objects',
                 '  and with usage of the bounding-box normalisation',
                 '- The pose-normalisation is not always optimal, which has a',
                 '  high impact on the results']
        messagebox.showinfo('Restrictions', "\n".join(lines))

    def clicked_button_fifth():
        lines = ['- Scaling-target-volume: reference-volume, that both ',
                 '                                          objects are scaled to before',
                 '                                          comparison (the higher the more',
                 '                                          accurate, the lower the more',
                 '                                          efficient)',
                 '- bounding box: additional normalisation-method to extend',
                 '                             the inertia-method',
                 '- pyvista-plotter additionally provides visualisation of the',
                 '   volume-similarity',
                 '- Density: every xth point is displayed',
                 '- Transparency: changes the appearance',
                 '- Transparency=0.0 starts the automatic-mode (optimized',
                 '                            for scaling target-volume of 100.000 and',
                 '                            resolution of 1920x1080)',
                 '- Recommended setting for matplotlib:',
                 '          Density 3, Transparency 0.05-0.2',
                 '          Density 2, Transparency 0.04-0.1',
                 '          Density 1, Transparency 0.01-0.02',
                 '- Recommended settings for plotly:',
                 '          Density 1-3',
                 '          Transparency 0.06-0.2',
                 '- Export-matrix: 3D numpy arrays of 0s(no object),',
                 '                           1s(only first object), 2s(only second object)',
                 '                           and 3s(both objects) as .csv files',
                 '- Export results: calculated similarity values as .txt file']
        messagebox.showinfo('Settings-information', "\n".join(lines))

    def main_process():
        start_all = time.time()

        if selected_illustration.get() == 'pyvista plotter' \
                or selected_illustration.get() == 'matplotlib' \
                or selected_illustration.get() == 'plotly':
            plot_time_text.configure(text='Time to plot:', font=('Aerial', 10))

        names.configure(text='')
        larger_text.configure(text='')
        convex_text.configure(text='')
        similarity_volume.configure(text='?')
        similarity_shape.configure(text='?')
        computation_time.configure(text='?')
        matrix_time.configure(text='?')
        export_time.configure(text='?')
        plot_time.configure(text='?')

        if selected_illustration.get() == 'plotly' or selected_illustration.get() == 'plotly(+export)':
            if var_1.get() > 3:
                messagebox.showinfo('Error', 'Density has to be between 1 and 3 for plotly-illustration!')
                return
        if tar_var.get() < 1000 or tar_var.get() > 1000000:
            messagebox.showinfo('Error', 'Target-volume has to be between 1.000 and 1.000.000')
            return

        if bool(file_1) and bool(file_2):
            # get file formats
            files = [file_1[0], file_2[0]]
            name_unused, format_1 = os.path.splitext(file_1[0])
            name_unused, format_2 = os.path.splitext(file_2[0])
            f = [format_1, format_2]

            # convert stl and/or csv objects to trimesh mesh-objects:
            trimesh_meshes, not_watertight = make_mesh(files, f)

            # calculate volume similarity and surface-area similarity:
            volume_similarity, larger_object = compute_vol_sim(trimesh_meshes)

            original_meshes = copy.deepcopy(trimesh_meshes)
            move(original_meshes)

            # compute final matrices for shape comparison and illustration:
            target_vol = tar_var.get()
            try:
                shape_similarity, final_matrices, final_meshes, time_matrix = \
                    process_meshes(trimesh_meshes, target_vol, check_check.get())

                # calculate the matrices for illustration:
                m_wmd = get_comparison_matrix(final_matrices)

                end_all = time.time()

                start = time.time()
                # export m_wmd:
                if check_decision.get():
                    export_matrix(m_wmd, str(basename_1[0]) + '-vs.-' + str(basename_2[0]) + ' (comparison_matrix)')

                # export results:
                if check_if.get():
                    with open(str(basename_1[0]) + '-vs.-' + str(basename_2[0]) + ' (similarity-results)' + '.txt',
                              'w') as text_file_results:
                        text_file_results.write(str(basename_1[0]) + '-vs.-' + str(basename_2[0]) + '\n\n')
                        if larger_object == 1:
                            text_file_results.write(str(basename_1[0]) + ' volume > '
                                                    + str(basename_2[0]) + ' volume\n')
                        elif larger_object == 2:
                            text_file_results.write(str(basename_1[0]) + ' volume < '
                                                    + str(basename_2[0]) + ' volume\n')
                        else:
                            text_file_results.write(str(basename_1[0]) + ' volume = '
                                                    + str(basename_2[0]) + ' volume\n')
                        if not_watertight == 1:
                            text_file_results.write(
                                '(The convex hull of one input-mesh had to be used for comparison\n')
                        elif not_watertight == 2:
                            text_file_results.write(
                                '(The convex hulls of both input-meshes had to be used for comparison\n')
                        text_file_results.write('\nVolume-similarity:    ' + str(round(volume_similarity, 4))
                                                + ' %\n')
                        text_file_results.write('Shape-similarity:    ' + str(round(shape_similarity, 4)))

                end = time.time()
                time_export = end - start

                # output of results and illustration:
                show = selected_illustration.get()
                names.configure(text=str(basename_1[0]) + '  vs.  ' + str(basename_2[0]))
                if larger_object == 1:
                    larger_text.configure(text=str(basename_1[0]) + ' volume > ' + str(basename_2[0]) + ' volume\n')
                elif larger_object == 2:
                    larger_text.configure(text=str(basename_1[0]) + ' volume < ' + str(basename_2[0]) + ' volume\n')
                else:
                    larger_text.configure(text=str(basename_1[0]) + ' volume = ' + str(basename_2[0]) + ' volume\n')
                if not_watertight == 1:
                    convex_text.configure('(the convex hull of one input-mesh had to be used for comparison)')
                elif not_watertight == 2:
                    convex_text.configure('(the convex hulls of both input-meshes had to be used for comparison)')
                similarity_volume.configure(text=str(round(volume_similarity, 4)) + ' %')
                similarity_shape.configure(text=str(round(shape_similarity, 4)) + ' %')
                computation_time.configure(text=str(round(end_all - start_all, 3)) + ' sec')
                matrix_time.configure(text=str(round(time_matrix, 3)) + ' sec')
                export_time.configure(text=str(round(time_export, 3)) + ' sec')

                dense = float(var_1.get())
                transpar = float(var_2.get())
                time_plot = illustrate(m_wmd, final_meshes, original_meshes, show, dense, transpar,
                                       shape_similarity, basename_1, basename_2)
                plot_time.configure(text=str(round(time_plot, 3)) + ' sec')

            except:
                if selected_illustration.get() == 'matplotlib' \
                        or selected_illustration.get() == 'matplotlib(+export png)':
                    messagebox.showinfo('Error', 'Scaling-target-volume too large for this computer '
                                                 'or too small for reasonable matplotlib-illustration')
                else:
                    messagebox.showinfo('Error', 'Scaling-target-volume too large for this computer')

        else:
            messagebox.showinfo('Error', 'Choose two objects to compare!')

    btn_1 = tk.Button(frame_00, text="Choose first object", bg="darkgrey", fg="black",
                      font=('Aerial', 10), command=clicked_button_first, activebackground='grey')
    btn_1.pack(fill='both', expand=True)
    btn_1.bind("<Enter>", lambda e: btn_1.config(bg='grey'))
    btn_1.bind("<Leave>", lambda e: btn_1.config(bg='darkgrey'))
    btn_2 = tk.Button(frame_02, text="Choose second object", bg="darkgrey", fg="black",
                      font=('Aerial', 10), command=clicked_button_second, activebackground='grey')
    btn_2.pack(fill='both', expand=True)
    btn_2.bind("<Enter>", lambda e: btn_2.config(bg='grey'))
    btn_2.bind("<Leave>", lambda e: btn_2.config(bg='darkgrey'))
    btn_3 = tk.Button(frame_60, text="information", font=('Aerial', 11), command=clicked_button_third,
                      fg='salmon', bg='white', relief='flat', activebackground='white', activeforeground='orangered')
    btn_3.pack(side='left', fill='both', expand=False)
    btn_3.bind("<Enter>", lambda e: btn_3.config(fg='orangered'))
    btn_3.bind("<Leave>", lambda e: btn_3.config(fg='salmon'))
    btn_4 = tk.Button(frame_70, text="restrictions", font=('Aerial', 11), command=clicked_button_fourth,
                      fg='salmon', bg='white', relief='flat', activebackground='white', activeforeground='orangered')
    btn_4.pack(side='left', fill='both', expand=False)
    btn_4.bind("<Enter>", lambda e: btn_4.config(fg='orangered'))
    btn_4.bind("<Leave>", lambda e: btn_4.config(fg='salmon'))
    btn_5 = tk.Button(frame_80, text="settings-information", font=('Aerial', 11), command=clicked_button_fifth,
                      fg='salmon', bg='white', relief='flat', activebackground='white', activeforeground='orangered')
    btn_5.pack(side='left', fill='both', expand=False)
    btn_5.bind("<Enter>", lambda e: btn_5.config(fg='orangered'))
    btn_5.bind("<Leave>", lambda e: btn_5.config(fg='salmon'))
    frame_100.pack_propagate(False)
    btn_go = tk.Button(frame_100, text="Start comparison", bg="lightblue", fg="black", font=('Aerial', 11),
                       command=main_process, cursor="hand2", activebackground='lightskyblue')
    # get the similarities from main_process
    btn_go.pack(fill='both', expand=True)
    btn_go.bind("<Enter>", lambda e: btn_go.config(bg='lightskyblue'))
    btn_go.bind("<Leave>", lambda e: btn_go.config(bg='lightblue'))

    window.columnconfigure(6, weight=1)
    window.rowconfigure(17, weight=1)

    window.mainloop()


if __name__ == "__main__":
    main()
