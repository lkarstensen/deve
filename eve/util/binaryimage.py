from dataclasses import dataclass

from ..vesseltree import Branch, VesselTree

from typing import List, NamedTuple

import numpy as np
import cv2

# check if copy is needed!
import copy
import numpy as np
from PIL import Image, ImageDraw, ImageOps


class Wall(NamedTuple):
    start: np.ndarray
    end: np.ndarray


@dataclass
class BinaryImage:
    image: Image
    spacing: float
    world_offset: np.ndarray
    dimension_to_omit: str

    # position of point and radius in maze coordinate system
    def draw_branch(self, centerline_branch: Branch):
        draw = ImageDraw.Draw(self.image)

        for centerline_point, radius in zip(
            centerline_branch.coordinates, centerline_branch.radii
        ):
            if self.dimension_to_omit == "x":
                point = np.array([centerline_point[1], centerline_point[2]])
            if self.dimension_to_omit == "z":
                point = np.array([centerline_point[0], centerline_point[1]])
            else:
                point = np.array([centerline_point[0], centerline_point[2]])
            position = self._coord_to_image(point)
            radius = np.floor(radius / self.spacing)

            circle_bb_low = tuple([coord - radius for coord in position])
            circle_bb_high = tuple([coord + radius for coord in position])
            draw.ellipse([circle_bb_low, circle_bb_high], fill=255)

    def _coord_to_image(self, point: np.ndarray):
        img_point = np.round((point - self.world_offset) / self.spacing, 0)
        return img_point

    def add_margin(self, margin_size=1):
        self.image = ImageOps.expand(self.image, border=margin_size, fill=0)
        self.world_offset -= self.spacing


def create_empty_binary_image_from_centerlinetree(
    centerline_tree: VesselTree, spacing: float, dimension_to_omit: str
) -> BinaryImage:
    axes_length = (
        centerline_tree.coordinate_space.high - centerline_tree.coordinate_space.low
    )
    if dimension_to_omit == "x":
        axes_length = [axes_length[1], axes_length[2]]
        world_offset = np.array(
            [
                centerline_tree.coordinate_space.low[1],
                centerline_tree.coordinate_space.low[2],
            ]
        )
    elif dimension_to_omit == "z":
        axes_length = [axes_length[0], axes_length[1]]
        world_offset = np.array(
            [
                centerline_tree.coordinate_space.low[0],
                centerline_tree.coordinate_space.low[1],
            ]
        )
    else:
        axes_length = [axes_length[0], axes_length[2]]
        world_offset = np.array(
            [
                centerline_tree.coordinate_space.low[0],
                centerline_tree.coordinate_space.low[2],
            ]
        )
    shape = np.ceil(np.array(axes_length) / spacing).astype(int)

    # margin left and right for each dimension
    shape += 2

    world_offset -= spacing

    # PIL automatically uses shape[0] for nb of cols (and vice versa)
    # find contours needs shape to be white
    zero_array = Image.new(mode="L", size=tuple(shape), color=0)
    binary_array = BinaryImage(
        zero_array,
        np.array(spacing, dtype=np.float32),
        np.array(world_offset, dtype=np.float32),
        dimension_to_omit,
    )

    return binary_array


def get_contours_from_binary_image(
    binary_image: BinaryImage, approx_margin=1
) -> np.ndarray:
    # make sure that there is margin around shape in order to find contours
    binary_image.add_margin()

    image_array = np.asarray(binary_image.image)
    cv_image = np.uint8(copy.copy(image_array))

    # find contours (for possible future cleaning use: RETR_TREE, hierarchy)
    # explanation on hierarchy:
    # https://docs.opencv.org/4.x/d9/d8b/tutorial_py_contours_hierarchy.html
    all_contours, _ = cv2.findContours(cv_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    approx_contours = []
    for c in all_contours:
        # approximate contours
        approx_c = cv2.approxPolyDP(c, approx_margin, True)
        approx_c = np.squeeze(approx_c)
        # basic cleaning: contours of size 2 are lines
        if approx_c.shape[0] >= 3:
            approx_contours.append(approx_c)

    real_world_contours = _contours_to_realworld(approx_contours, binary_image)

    return real_world_contours


def _contours_to_realworld(
    contours: np.ndarray, binary_image: BinaryImage
) -> np.ndarray:
    realworld_contours = []
    for c in contours:
        realworld_c = c * binary_image.spacing + binary_image.world_offset
        realworld_contours.append(realworld_c)

    return realworld_contours


def contours_to_walls(contours: np.ndarray) -> List[Wall]:
    list_of_walls = []
    for c in contours:
        nb_points = c.shape[0]
        for i in range(nb_points):
            start = np.array([c[i, 0], c[i, 1]])
            # connect last and first point
            end = np.array([c[(i + 1) % nb_points, 0], c[(i + 1) % nb_points, 1]])

            list_of_walls.append(Wall(start, end))

    return list_of_walls


def create_walls_from_vessel_tree(
    vessel_tree: VesselTree,
    dimension_to_omit: str,
    pixel_spacing=0.2,
    contour_approx_margin=2.0,
) -> List[Wall]:
    binary_image = create_empty_binary_image_from_centerlinetree(
        centerline_tree=vessel_tree,
        spacing=pixel_spacing,
        dimension_to_omit=dimension_to_omit,
    )

    for branch in vessel_tree.branches:
        binary_image.draw_branch(branch)

    contours = get_contours_from_binary_image(
        binary_image, approx_margin=contour_approx_margin
    )

    return contours_to_walls(contours)
