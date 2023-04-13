import slicer
import os
import pickle
import argparse

try:
    import numpy as p
except ImportError:
    slicer.util.pip_install('numpy==1.20.3')

try:
    import vtk
except ImportError:
    slicer.util.pip_install('vtk==9.0.3')

from vtk.util.numpy_support import vtk_to_numpy

try:
    import PIL
except ImportError:
    slicer.util.pip_install('Pillow==8.3.1')

from PIL import Image

from geometry_image.tools.vtk_tools import *


def compute_spherical_area(p1, p2, p3):
    # length of the sides of the triangle
    a = np.arccos(np.diag(np.dot(p2.T, p3)))
    b = np.arccos(np.diag(np.dot(p1.T, p3)))
    c = np.arccos(np.diag(np.dot(p1.T, p2)))
    s = (a + b + c) / 2

    # L'Huilier's Theorem
    # tand(E/4)^2 = tan(s/2)tan( (s-a)/2 )tan( (s-b)/2 )tan( (s-c)/2 )
    E = np.tan(s/2) * np.tan((s - a)/2) * np.tan((s - b)/2) * np.tan((s - c)/2)
    A = np.real(4 * np.arctan(np.lib.scimath.sqrt(E)))
    return A


def perform_spherial_planar_sampling(pos_shpere: np.ndarray, sampling_type: str) -> np.ndarray:
    """project sampling location from sphere to a square.
    This is used to produced spherical geometry images.
    The sampling is first projected onto an octahedron and then unfolded on a square.

    :param pos_shpere: ndarray with columns containing the (x, y, z) coordinates on the sphere
    :param sampling_type: 'area' or 'genomic'
    :return: pos_plane: ndarray with columns cotaining (x, y) coordinates on the plane
    """
    a = range(-1, 2, 2)  # [-1 1]
    grid = np.mgrid[a, a, a].reshape(3, 8)

    # Get anchor points for each quadrant
    anchor_2d = []
    anchor_3d = []
    for quad in range(8):
        a3d = np.diag(grid[:, quad])  # Creating anchor points (x, 0, 0), (0, y, 0), (0, 0, z)
        anchor_3d.append(np.roll(a3d, 1, axis=1))  # moving z vertex to the top

        # Create corresponding 2d anchor points for the plane
        x = grid[0, quad]
        y = grid[1, quad]
        z = grid[2, quad]
        if z > 0:
            # Collapse (0, 0, z) to  (0, 0) if z is positive
            a2d = np.array([[0, 0], [x, 0], [0, y]]).T
        else:
            # Otherwise collapse (0, 0, z) to (x, y)
            a2d = np.array([[x, y], [x, 0], [0, y]]).T
        anchor_2d.append(a2d)

    pos_shpere_sign = (pos_shpere >= 0) * 2 - 1
    posw = np.zeros((2, pos_shpere.shape[1]))
    # project sphere at each quadrant individually
    for quad in range(0, 8):
        a2d = anchor_2d[quad]
        a3d = anchor_3d[quad]
        # find indices of the points that is in this quadrant
        idx = np.where((pos_shpere_sign == grid[:, quad:quad+1]).all(axis=0))[0]
        points = pos_shpere[:, idx]
        n = points.shape[1]
        if sampling_type == "area":
            # find the area of 3 small triangles
            p1 = a3d[:, 0:1].repeat(n, axis=1)
            p2 = a3d[:, 1:2].repeat(n, axis=1)
            p3 = a3d[:, 2:3].repeat(n, axis=1)
            a1 = compute_spherical_area(points, p2, p3)
            a2 = compute_spherical_area(points, p1, p3)
            a3 = compute_spherical_area(points, p1, p2)
            # barycentric coordinates
            a = a1 + a2 + a3
            a1 /= a
            a2 /= a
            a3 /= a
        else:
            raise Exception("Unknown sampling type!!!")
        posw[:, idx] = np.dot(a2d[:, 0:1], a1[np.newaxis, :]) + \
                       np.dot(a2d[:, 1:2], a2[np.newaxis, :]) + \
                       np.dot(a2d[:, 2:3], a3[np.newaxis, :])
    return posw


def pad_2d_image(im2):
    im_pad_right = np.flip(np.flip(im2[:, im2.shape[1]//2:], axis=1), axis=0)
    im_pad_left = np.flip(np.flip(im2[:, 0:im2.shape[1]//2], axis=1), axis=0)
    im_pad_bottom = np.flip(np.flip(im2[im2.shape[0]//2:, :], axis=0), axis=1)
    im_pad_top = np.flip(np.flip(im2[0:im2.shape[0]//2, :], axis=0), axis=1)
    im_pad_left_top = im2[im2.shape[0]//2:, im2.shape[1]//2:]
    im_pad_right_bottom = im2[0:im2.shape[0]//2, 0:im2.shape[0]//2]
    im_pad_left_bottom = im2[0:im2.shape[0]//2, im2.shape[1]//2:]
    im_pad_right_top = im2[im2.shape[0]//2:, 0:im2.shape[1]//2]
    im_padded = np.concatenate([im_pad_top, im2, im_pad_bottom], axis=0)
    left_pad = np.concatenate([im_pad_left_top, im_pad_left, im_pad_left_bottom], axis=0)
    right_pad = np.concatenate([im_pad_right_top, im_pad_right, im_pad_right_bottom], axis=0)
    im_padded = np.concatenate([left_pad, im_padded, right_pad], axis=1)
    return im_padded


def render_2d_image(coord, z, res="100"):
    from scipy.interpolate import griddata
    coord = coord * 0.5 + 0.5
    res = complex(str(int(res)//2) + "j")
    grid_x, grid_y = np.mgrid[0:1:res, 0:1:res]
    return griddata(coord, z, (grid_x, grid_y), method="nearest")


def get_flat_coordinates(sphere_template):
    if os.path.exists(sphere_template.split(".")[0] + "_flat.p"):
        posw = pickle.load(open(sphere_template.split(".")[0] + "_flat.p", "rb"))
    else:
        poly_data = ReadPolyData(sphere_template)
        proj_shpere = vtk_to_numpy(poly_data.GetPoints().GetData()).T
        posw = perform_spherial_planar_sampling(proj_shpere, "area")
        pickle.dump(posw, open(sphere_template.split(".")[0] + "_flat.p", "wb"))
    return posw


def normalize(image_array):
    _min = np.min(image_array)
    _max = np.max(image_array)
    image_array = np.floor((image_array - _min) /
                           (_max - _min) * 256)
    return image_array.astype(np.uint8)


def sgim_sampling_wrapper(args):
    # Process sphere template to get 2D coordinates
    posw = get_flat_coordinates(args["sphere_template"])
    feature = np.genfromtxt(args["feature_map"])

    # Render 2D image with 2D coordinates and feature map
    image_array = render_2d_image(posw.T, feature, args["resolution"])
    image_array = pad_2d_image(image_array)
    image_array = normalize(image_array)
    im = Image.fromarray(image_array)
    im.save(args["output"])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--sphere_template", type=str, required=True,
                        help="The sphere template to map feature")
    parser.add_argument("-f", "--feature_map", type=str, required=True,
                        help="feature corresponding to sphere template")
    parser.add_argument("-r", "--resolution", type=str, default="512",
                        help="Resolution of output 2D image")
    parser.add_argument("-o", "--output", type=str, help="output image")
    args = parser.parse_args()
    args = vars(args)
    sgim_sampling_wrapper(args)
