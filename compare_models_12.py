# Frederik Frerck 25.09.2022


import os
import csv
import math
import copy
import time
# import meshio
import trimesh
import argparse
import numpy as np
import pandas as pd
import pyvista as pv
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from stl import mesh  # numpy-stl library, not stl library!!
from functools import reduce
from tkinter import filedialog
from pyntcloud import PyntCloud
from trimesh import transformations
from trimesh import convex
from trimesh import inertia
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
            # rm_shape = cur_vol_int.shape
            # print('Matrix shape:', rm_shape)
            # cur_vol_int = cur_vol.astype(np.uint8)
            resulting_matrices.append(cur_vol_int)
        return resulting_matrices


# convert two csv or stl objects into two trimesh-meshes
# input: list of two objects as csv or stl file and list of file-formats as string
def make_mesh(input_file_paths, file_form):
    initial_meshes = []
    incorrect_mesh = 0
    for i, input_file_path in enumerate(input_file_paths):
        if file_form[i] == 'csv':
            # .csv to .obj
            cloud = pd.read_csv(input_file_path, skiprows=1, header=None)
            cloud.columns = ["x", "y", "z"]
            cloud = PyntCloud(cloud)

            # reconstruct point cloud to mesh with pyvista:
            # point_array = np.asarray(cloud.points)
            # points = pv.wrap(point_array)
            # volume = points.delaunay_3d()
            # shell = volume.extract_geometry()
            # shell.save("output.stl")
            # netz = points.reconstruct_surface()
            # netz.save("output.stl")
            # a_mesh = trimesh.load("output.stl", file_type='stl')
            # trimesh.repair.fill_holes(a_mesh)
            # trimesh.repair.fix_inversion(a_mesh, multibody=False)
            # trimesh.repair.fix_normals(a_mesh, multibody=False)

            # first convex hull:
            convex_hull_id = cloud.add_structure("convex_hull")
            convex_hull = cloud.structures[convex_hull_id]
            cloud.mesh = convex_hull.get_mesh()
            cloud.to_file("output.obj", also_save=["mesh"])
            # .obj to .stl
            # loaded_mesh = trimesh.load("output.obj", file_type='obj')  # without conversion to stl
            # mesh_io = meshio.read("output.obj")
            # meshio.write_points_cells("output.stl", mesh_io.points, mesh_io.cells)
            loaded_mesh = trimesh.load("output.obj", file_type='obj')
            # second convex hull:
            a_mesh = convex.convex_hull(loaded_mesh)

            initial_meshes.append(a_mesh)
            os.remove('output.obj')
            # os.remove('output.stl')
        else:
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
    bounding_box = bounding_box * 2  # 1 would be boundary box
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
        # print('Volume:', vol_i)
        if target_volume > vol_i:
            percentage = vol_i / target_volume
            while round(percentage, 5) < 1.0:
                # factor = 1 + (0.01 / percentage)
                factor = 1 + (-math.log(percentage) / 10)
                initial_meshes[i].apply_scale(factor)
                # compute volume of the smaller object:
                vol_i = initial_meshes[i].volume
                # new volume-similarity for loop:
                percentage = vol_i / target_volume
        elif target_volume < vol_i:
            percentage = target_volume / vol_i
            while round(percentage, 5) < 1.0:
                # factor = 1 - (0.01 / percentage)
                factor = 1 - (-math.log(percentage) / 10)
                if factor <= 0.00001:
                    factor = 0.01
                initial_meshes[i].apply_scale(factor)
                # compute volume of the smaller object:
                vol_i = initial_meshes[i].volume
                # new volume-similarity for loop:
                percentage = target_volume / vol_i


# compute the volume-similarity, the surface-area-similarity and the center-of-mass of two trimesh-meshes
def compute_sims(initial_meshes):
    vol_1 = initial_meshes[0].volume
    vol_2 = initial_meshes[1].volume
    larger = 0
    if vol_1 < vol_2:
        similarity_1 = (vol_1 / vol_2) * 100
        larger = 2
    elif vol_1 > vol_2:
        similarity_1 = (vol_2 / vol_1) * 100
        larger = 1
    else:
        similarity_1 = 100

    area_1 = initial_meshes[0].area
    area_2 = initial_meshes[1].area
    if area_1 <= area_2:
        similarity_2 = (area_1 / area_2) * 100
    else:
        similarity_2 = (area_2 / area_1) * 100

    return similarity_1, similarity_2, larger


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
    # in m_with_distinction: 1s and 2s is where the objects differ, 3s is where they match/overlap
    return m_with_distinction


# calculate the similarity of two matrices representing objects and containing 1s for material and 0s for no material
# input: list of matrices as lists of lists of lists (3D)
def shape_comparison(matrices):
    # matrices contain the finally shifted and scaled objects
    m = matrices[0] + matrices[1]  # 1s is where the objects differ, 2s is where they match/overlap

    flat_array_1 = matrices[0].flatten()
    vol_1_list = flat_array_1.tolist()
    count_1s_in_1 = vol_1_list.count(1)
    # print('Skalierungsvolumen Objekt 1:', count_1s_in_1)
    flat_array_2 = matrices[1].flatten()
    vol_2_list = flat_array_2.tolist()
    count_1s_in_2 = vol_2_list.count(1)
    # print('Skalierungsvolumen Objekt 2:', count_1s_in_2)
    point_difference = count_1s_in_2 - count_1s_in_1
    # print('Differenz der Punkt-Anzahl der skalierten Objekte:', point_difference)
    reference = (count_1s_in_1 + count_1s_in_2) / 2
    flat_array = m.flatten()
    vol_list = flat_array.tolist()
    count_2s = vol_list.count(2)

    # count = count_positive + count_negative
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
    matrices_bb = mtm.mesh_to_matrix(stl_meshes_bb, shape_bb)
    shape_similarity_bb_new = shape_comparison(matrices_bb)

    # p_m_1 = pv.wrap(bb_meshes[0])
    # p_m_2 = pv.wrap(bb_meshes[1])
    # pl = pv.Plotter()
    # _ = pl.add_mesh(p_m_1, color='slateblue', opacity=0.5)
    # _ = pl.add_mesh(p_m_2, color='skyblue', opacity=0.5)
    # pl.add_title(str(shape_similarity_bb_new), font_size=6, color='black')
    # pl.set_background('white')
    # pl.show()

    if turn_back == 1:
        qx = transformations.quaternion_about_axis(-degs[0], [1, 0, 0])
        qy = transformations.quaternion_about_axis(-degs[1], [0, 1, 0])
        qz = transformations.quaternion_about_axis(-degs[2], [0, 0, 1])
        q = transformations.quaternion_multiply(qz, qy)
        q = transformations.quaternion_multiply(q, qx)
        r = transformations.quaternion_matrix(q)
        bb_meshes[1].apply_transform(r)

    return shape_similarity_bb_new


# execute the normalisation of two trimesh-meshes based on the inertia-axes and the centers of mass
def process_meshes(the_meshes, target, if_bb):
    # compute the center of mass
    center_of_mass_1 = the_meshes[0].center_mass
    center_of_mass_2 = the_meshes[1].center_mass

    # shift meshes to the same center of mass:
    the_meshes[0].apply_translation(-center_of_mass_1)
    the_meshes[1].apply_translation(-center_of_mass_2)

    bounding_box_meshes = copy.deepcopy(the_meshes)
    scale(bounding_box_meshes, target)
    shape_similarity_bb = 0

    # calculate the principal axis of inertia:
    inertia_axes_0, inertia_axes_1 = calculate_rotation(the_meshes)
    # print('Achsen:', inertia_axes_0, inertia_axes_1)

    turned_meshes = copy.deepcopy(the_meshes)

    # rotate meshes:
    if inertia_axes_0[0][0] != 0 or inertia_axes_0[0][1] != 0:
        rotate(the_meshes[0], inertia_axes_0[0], [0, 0, 1])
        rotate(turned_meshes[0], inertia_axes_0[0], [0, 0, 1])

    if inertia_axes_1[0][0] != 0 or inertia_axes_1[0][1] != 0:
        rotate(the_meshes[1], inertia_axes_1[0], [0, 0, 1])
        rotate(turned_meshes[1], inertia_axes_1[0], [0, 0, -1])
    elif inertia_axes_1[0][0] == 0 and inertia_axes_1[0][1] == 0:
        rotate(turned_meshes[1], inertia_axes_1[0], [1, 0, 0])
        rotate(turned_meshes[1], inertia_axes_1[0], [0, 0, -1])

    inertia_axes_0, inertia_axes_1 = calculate_rotation(the_meshes)
    # inertia_axes_turned_0, inertia_axes_turned_1 = calculate_rotation(turned_meshes)
    # print('Achsen neu:', inertia_axes_0, inertia_axes_1)
    # print('[0,0,1]:', inertia_axes_turned_0[0])

    # p_m_1 = pv.wrap(the_meshes[0])
    # p_m_2 = pv.wrap(the_meshes[1])
    # pl = pv.Plotter()
    # _ = pl.add_mesh(p_m_1, color='lightskyblue', opacity=0.4)
    # _ = pl.add_mesh(p_m_2, color='darkblue', opacity=0.4)
    # pl.set_background('white')
    # pl.show()

    # Rotation des ersten Objektes um die z-Achse
    # Die neuen Trägheitsachsen zwei und drei liegen jetzt in der x-y-Ebene (z-Parameter = 0)
    if inertia_axes_0[1][0] != 0:  # inertia_axes_0[1][2] ist hier immer gleich Null oder?
        normalized_0_1 = inertia_axes_0[1] / np.linalg.norm(inertia_axes_0[1])  # Neuer Betrag = 1 f"ur Einheitskreis
        if normalized_0_1[0] < 0:
            alpha = (math.asin(normalized_0_1[1]) - (math.pi / 2)) / math.pi * 180  # Winkelberechnung
        else:
            alpha = (-1) * (math.asin(normalized_0_1[1]) - (math.pi / 2)) / math.pi * 180
        q = transformations.quaternion_about_axis(np.radians(alpha), inertia_axes_0[0])  # Bestimmung der Quaternationen
        r = transformations.quaternion_matrix(q)  # zur Durchführung der Rotation mit Transformationsmatrix
        the_meshes[0].apply_transform(r)
        turned_meshes[0] = the_meshes[0]

    # Rotation des zweiten Objektes um die z-Achse -> Achtung einmal um 180 Grad gedreht!
    inertia_axes_0, inertia_axes_1 = calculate_rotation(the_meshes)
    if inertia_axes_1[1][0] != 0:
        normalized_1_1 = inertia_axes_1[1] / np.linalg.norm(inertia_axes_1[1])
        if normalized_1_1[0] < 0:
            alpha = (math.asin(normalized_1_1[1]) - (math.pi / 2)) / math.pi * 180
        else:
            alpha = (-1) * (math.asin(normalized_1_1[1]) - (math.pi / 2)) / math.pi * 180
        q = transformations.quaternion_about_axis(np.radians(alpha), inertia_axes_1[0])
        # Abweichung von inertia_axes_1[0] und [0,0,1] messen
        r = transformations.quaternion_matrix(q)
        the_meshes[1].apply_transform(r)

    inertia_axes_turned_0, inertia_axes_turned_1 = calculate_rotation(turned_meshes)
    if inertia_axes_turned_1[1][0] != 0:
        normalized_turned_1_1 = inertia_axes_turned_1[1] / np.linalg.norm(inertia_axes_turned_1[1])
        if normalized_turned_1_1[0] < 0:
            alpha = (math.asin(normalized_turned_1_1[1]) - (math.pi / 2)) / math.pi * 180
        else:
            alpha = (-1) * (math.asin(normalized_turned_1_1[1]) - (math.pi / 2)) / math.pi * 180
        q = transformations.quaternion_about_axis(np.radians(alpha), inertia_axes_turned_1[0])  # v
        r = transformations.quaternion_matrix(q)
        turned_meshes[1].apply_transform(r)

    the_meshes_rotated = copy.deepcopy(the_meshes)
    q = transformations.quaternion_about_axis(np.radians(180), inertia_axes_turned_1[0])
    r = transformations.quaternion_matrix(q)
    the_meshes_rotated[1].apply_transform(r)
    turned_meshes_rotated = copy.deepcopy(turned_meshes)
    q = transformations.quaternion_about_axis(np.radians(180), inertia_axes_turned_1[0])
    r = transformations.quaternion_matrix(q)
    turned_meshes_rotated[1].apply_transform(r)

    scale(the_meshes, target)  # Funktioniert die Rotation besser, wenn die Objekte größer sind?
                               # Werden die Volumina in dem Fall von der Rotation beeinflusst?
    com_the_1 = the_meshes[0].center_mass
    com_the_2 = the_meshes[1].center_mass
    the_meshes[0].apply_translation(-com_the_1)
    the_meshes[1].apply_translation(-com_the_2)
    shift = offset(the_meshes)
    the_meshes[0].apply_translation(shift)
    the_meshes[1].apply_translation(shift)

    scale(turned_meshes, target)
    com_turned_1 = turned_meshes[0].center_mass
    com_turned_2 = turned_meshes[1].center_mass
    turned_meshes[0].apply_translation(-com_turned_1)
    turned_meshes[1].apply_translation(-com_turned_2)
    shift = offset(turned_meshes)
    turned_meshes[0].apply_translation(shift)
    turned_meshes[1].apply_translation(shift)

    scale(the_meshes_rotated, target)
    com_rotated_1 = the_meshes_rotated[0].center_mass
    com_rotated_2 = the_meshes_rotated[1].center_mass
    the_meshes_rotated[0].apply_translation(-com_rotated_1)
    the_meshes_rotated[1].apply_translation(-com_rotated_2)
    shift = offset(the_meshes_rotated)
    the_meshes_rotated[0].apply_translation(shift)
    the_meshes_rotated[1].apply_translation(shift)

    scale(turned_meshes_rotated, target)
    com_turned_rotated_1 = turned_meshes_rotated[0].center_mass
    com_turned_rotated_2 = turned_meshes_rotated[1].center_mass
    turned_meshes_rotated[0].apply_translation(-com_turned_rotated_1)
    turned_meshes_rotated[1].apply_translation(-com_turned_rotated_2)
    shift = offset(turned_meshes_rotated)
    turned_meshes_rotated[0].apply_translation(shift)
    turned_meshes_rotated[1].apply_translation(shift)

    if if_bb == 1:
        for two_times in range(2):
            transform_1, extends_1 = trimesh.bounds.oriented_bounds(bounding_box_meshes[0])
            transform_2, extends_2 = trimesh.bounds.oriented_bounds(bounding_box_meshes[1])
            bounding_box_meshes[0].apply_transform(-transform_1)
            bounding_box_meshes[1].apply_transform(-transform_2)

        meshes_bb = trimesh_to_stl_mesh(bounding_box_meshes)
        shape_bb = calculate_shape(meshes_bb)
        mtm = MeshToMatrix()
        matrices_bb = mtm.mesh_to_matrix(meshes_bb, shape_bb)
        shape_similarity_bb_old = shape_comparison(matrices_bb)
        degrees = [0, 0, 0]
        remember_degrees = [0, 0, 0]
        # 24 possibilities (cube):
        for x in range(2):
            degrees[0] = np.radians(90 * x)  # math.pi * 0.5 * x
            for y in range(3):
                degrees[1] = np.radians(-90 * y)
                for z in range(4):
                    degrees[2] = np.radians(90 * z)
                    shape_similarity_bb_new = bounding_box_comparison(bounding_box_meshes, degrees, 1)
                    if shape_similarity_bb_new > shape_similarity_bb_old:
                        shape_similarity_bb_old = shape_similarity_bb_new
                        remember_degrees = [degrees[0], degrees[1], degrees[2]]
        shape_similarity_bb = bounding_box_comparison(bounding_box_meshes, remember_degrees, 0)

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
    mtm_time = end - start
    shape_similarity = shape_comparison(final_matrices)
    shape_similarity_turned = shape_comparison(final_matrices_turned)
    shape_similarity_rotated = shape_comparison(final_matrices_rotated)
    shape_similarity_rotated_turned = shape_comparison(final_matrices_rotated_turned)
    print()
    print(round(shape_similarity, 4))
    print(round(shape_similarity_turned, 4))
    print(round(shape_similarity_rotated, 4))
    print(round(shape_similarity_rotated_turned, 4))
    print(round(shape_similarity_bb, 4))

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

    trimesh.exchange.export.export_mesh(the_meshes[0], 'output_1.stl', file_type='stl')
    trimesh.exchange.export.export_mesh(the_meshes[1], 'output_2.stl', file_type='stl')

    return shape_similarity, final_matrices, the_meshes, mtm_time


# plot a similarity-matrix containing 0s, 1s, 2s and 3s with plotly
def plot_diff(m, dense, transparency, title):
    x_light = math.ceil(m.shape[0] / 2)
    y_light = math.ceil(m.shape[1] / 2)
    z_light = math.ceil(m.shape[2] / 2)
    light_matrix = np.zeros((x_light, y_light, z_light))
    for index, val in np.ndenumerate(m):
        if index[0] % dense == 0 and index[1] % dense == 0 and index[2] % dense == 0:
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
        surface_count=10  # needs to be a large number for good volume rendering
        # like transparency for plot_differences
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
    count_1 = 0
    count_2 = 0
    count_3 = 0
    for index, val in np.ndenumerate(m):
        if index[0] % dense == 0 and index[1] % dense == 0 and index[2] % dense == 0:
            if val == 1:
                one = 1
                count_1 += 1
                x_one.append(index[0])
                y_one.append(index[1])
                z_one.append(index[2])
            elif val == 2:
                two = 1
                count_2 += 1
                x_two.append(index[0])
                y_two.append(index[1])
                z_two.append(index[2])
            elif val == 3:
                count_3 += 1
                x_three.append(index[0])
                y_three.append(index[1])
                z_three.append(index[2])

    print('Number of 1s:', count_1)
    print('Number of 2s:', count_2)
    print('Number of 3s:', count_3)

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
def illustrate(similarity_1, larger, similarity_2, similarity_3, m_wmd, final_meshes,
               export_names, density=3, transparency=0.0, show=1, expo=0):
    plot_time = 0
    if transparency < 0.0 or transparency > 1:
        raise Exception('Transparency has to be between 0.0 and 1')
    if show < 0 or show > 3:
        raise Exception('show has to be 1 for yes or 0 for no')

    if larger == 1:
        print(str(export_names[0]) + ' volume > '
              + str(export_names[1]) + ' volume')
    elif larger == 2:
        print(str(export_names[0]) + ' volume < '
              + str(export_names[1]) + ' volume')
    else:
        print(str(export_names[0]) + ' volume = '
              + str(export_names[1]) + ' volume')
    similarity_1 = round(similarity_1, 4)
    similarity_2 = round(similarity_2, 4)
    similarity_3 = round(similarity_3, 4)
    print('     VOLUME-SIMILARITY:      ', similarity_1, '%')  # add, which object is larger/smaller
    print('     SURFACE-AREA-SIMILARITY:', similarity_2, '%')
    print('     SHAPE-SIMILARITY:       ', similarity_3, '%')
    if show == 2:
        print('Illustration with matplotlib:')
        print('default density: 3                           set density:', density)
        print('default transparency: 0.0 (automatic-mode)   set transparency:', transparency)
        title = 'Representation of the two normalised objects, scaled to the same volume'
        start = time.time()
        plot_differences(m_wmd, density, transparency, title, similarity_3)  # comparison with scaling
        if expo == 1:
            plt.savefig(str(export_names[0]) + '-vs.-' + str(export_names[1])
                        + ' (matplotlib-illustration)' + '.png')
        end = time.time()
        plot_time = end - start
        plt.show()
    elif show == 3:
        print('Illustration with plotly:')
        print('default density: 3                 set density:', density)
        print('default transparency: 0.0 (=0.1)   set transparency:', transparency)
        title = 'Representation of the two normalised objects, scaled to the same volume'
        start = time.time()
        figure = plot_diff(m_wmd, density, transparency, title)  # plotly go.Figure
        if expo == 1:
            figure.write_html(str(export_names[0]) + '-vs.-' + str(export_names[1])
                              + ' (plotly-illustration)' + '.html', auto_open=True)
        else:
            figure.show()
        end = time.time()
        plot_time = end - start
    elif show == 1:  # plot with the pyvista Plotter
        print('Illustration with pyvista plotter')
        start = time.time()
        p_m_1 = pv.wrap(final_meshes[0])
        p_m_2 = pv.wrap(final_meshes[1])
        pl = pv.Plotter(window_size=[800, 600])
        _ = pl.add_mesh(p_m_1, color='slateblue', opacity=0.5)  # skyblue and slateblue
        _ = pl.add_mesh(p_m_2, color='skyblue', opacity=0.5)  # lightskyblue and darkblue
        pl.add_title('Representation of the two normalised objects, scaled to the same volume',
                     font_size=6, color='black')
        pl.set_background('white')
        pl.camera.zoom(0.9)
        end = time.time()
        plot_time = end - start
        if expo == 1:
            pl.show(screenshot=str(export_names[0]) + '-vs.-' + str(export_names[1]) + ' (pyvista-illustration).png')
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


# check file formats to be .stl or .csv
def file_format(files):
    f = []
    for file in files:
        filename, ext = os.path.splitext(file)
        if ext.lower() == '.stl':
            f.append('stl')
        elif ext.lower() == '.csv':
            f.append('csv')
        else:
            raise ValueError('Wrong file format, only stl and csv allowed')
    return f


def main():
    parser = argparse.ArgumentParser(description='Compute similarity of two 3d objects')
    parser.add_argument('input', nargs='*')
    parser.add_argument('--scale', type=int, default=100000, help='Target volume for the scaling')
    parser.add_argument('--density', type=int, default=3, help='Density of points in 3D plot, 1 displays all points')
    parser.add_argument('--transparency', type=float, default=0.0, help='Transparency-parameter for matplotlib- and '
                                                                        'plotly-illustration; 0.0 = automatic-mode')
    parser.add_argument('--show', type=int, default=1, help='Illustrate the results? '
                                                            '1=pyvista-plotter 2=matplotlib 3=plotly 0=no illustration')
    parser.add_argument('--export', type=int, default=0, help='Export comparison matrices and results? 1=yes, 0=no')
    parser.add_argument('--boundingbox', type=int, default=1, help='Use the bounding box pose-normalisation '
                                                                   'additionally to the inertia-alignment? 1=yes, 0=no')

    args = parser.parse_args()

    # if args.scale < 100 or args.scale > 1000000:
    #     raise Exception('Target-scaling-volume has to be between 100 and 1.000.000 mm^3')
    if args.show == 1:
        if args.density < 1 or args.density > 5:
            raise Exception('Density-parameter has to be integer between 1 and 5 for matplotlib-illustration')
    elif args.show == 2:
        if args.density < 1 or args.density > 3:
            raise Exception('Density-parameter has to be integer between 1 and 3 for plotly-illustration')
    if args.transparency < 0.0 or args.transparency > 0.3:
        raise Exception('Transparency-parameter has to be between 0.00 and 0.30')
    if args.show < 0 or args.show > 3:
        raise Exception('Show-parameter has to be integer between 0 and 3')
    if args.export < 0 or args.export > 1:
        raise Exception('Export-parameter has to be 0 or 1')
    if args.boundingbox < 0 or args.boundingbox > 1:
        raise Exception('Bounding-Box-parameter has to be 0 or 1')

    # find file formats
    f = file_format(args.input)

    # open directory in case of no command line arguments
    if args.input is None or len(args.input) != 2:
        file_1 = filedialog.askopenfilename(initialdir=os.path.dirname(os.path.realpath("__file__")),
                                            title="Select a File", filetypes=(("stl files", "*.stl*"),
                                                                              ("csv files", "*.csv*")))
        file_2 = filedialog.askopenfilename(initialdir=os.path.dirname(os.path.realpath("__file__")),
                                            title="Select a File", filetypes=(("stl files", "*.stl*"),
                                                                              ("csv files", "*.csv*")))
        base_names = [os.path.basename(file_1), os.path.basename(file_2)]
        files = [file_1, file_2]
        f = file_format(files)
    else:
        base_names = []
        for file in args.input:
            base_names.append(os.path.basename(file))
        files = args.input

    start_all = time.time()
    # convert stl and/or csv objects to trimesh mesh-objects:
    trimesh_meshes, not_watertight = make_mesh(files, f)

    # calculate volume similarity and surface-area similarity:
    volume_similarity, surface_area_similarity, larger = compute_sims(trimesh_meshes)

    # compute final matrices for shape comparison and illustration:
    shape_similarity, final_matrices, final_meshes, time_matrix = \
        process_meshes(trimesh_meshes, args.scale, args.boundingbox)

    # calculate the matrix for illustration:
    m_wmd = get_comparison_matrix(final_matrices)

    end_all = time.time()

    # export m_wmd_1, m_og and results:
    time_export = 0
    if args.export == 1:
        start = time.time()
        export_matrix(m_wmd, str(base_names[0]) + '-vs.-' + str(base_names[1])
                      + ' (comparison_matrix)')
        end = time.time()
        time_export = end - start
        with open(str(base_names[0]) + '-vs.-' + str(base_names[1]) + ' (similarity-results)' + '.txt',
                  'w') as text_file_results:
            text_file_results.write(str(base_names[0]) + '-vs.-' + str(base_names[1]) + '\n\n')
            if larger == 1:
                text_file_results.write(str(base_names[0]) + ' volume > '
                                        + str(base_names[1]) + ' volume\n')
            elif larger == 2:
                text_file_results.write(str(base_names[0]) + ' volume < '
                                        + str(base_names[1]) + ' volume\n')
            else:
                text_file_results.write(str(base_names[0]) + ' volume = '
                                        + str(base_names[1]) + ' volume\n')
            if not_watertight == 1:
                text_file_results.write(
                    '(The convex hull of one input-mesh had to be used for comparison\n')
            elif not_watertight == 2:
                text_file_results.write(
                    '(The convex hulls of both input-meshes had to be used for comparison\n')
            text_file_results.write('\nVolume-similarity:    ' + str(round(volume_similarity, 4))
                                    + ' %\n')
            text_file_results.write('Shape-similarity:    ' + str(round(shape_similarity, 4)))

    # output of results and illustration:
    time_plot = illustrate(volume_similarity, larger, surface_area_similarity, shape_similarity, m_wmd, final_meshes,
                           base_names, args.density, args.transparency, args.show, args.export)

    # similarity_area = 0
    area_1 = final_meshes[0].area
    area_2 = final_meshes[1].area
    if area_1 <= area_2:
        similarity_area = (area_1 / area_2) * 100
    else:
        similarity_area = (area_2 / area_1) * 100
    # print('Oberflächeninhaltsähnlichkeit:', round(similarity_area, 4))

    print('time to compute:', round(end_all - start_all, 3), 'sec')
    print('time to "mesh_to_matrix":', round(time_matrix, 3), 'sec')
    if args.export == 1:
        print('time to plot and export:', round(time_plot, 3), 'sec')
        print('time to export matrices and results:', round(time_export, 3), 'sec')
    else:
        print('time to plot:', round(time_plot, 3), 'sec')
    if not_watertight == 1:
        print('The convex hull of one input-mesh had to be used for comparison')
    elif not_watertight == 2:
        print('The convex hulls of both input-meshes had to be used for comparison')
    print()


if __name__ == "__main__":
    main()
