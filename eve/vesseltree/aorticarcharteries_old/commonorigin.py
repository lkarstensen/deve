from typing import Tuple, List

from ..branch import Branch
from ..util.cubichermitesplines import CHSPoint, chs_point_normal, chs_to_cl_points

import numpy as np


def common_origin_VI(
    start: Tuple[float, float, float],
    resolution: float,
    rng: np.random.Generator = None,
) -> Tuple[Branch, List[CHSPoint]]:
    rng = rng or np.random.default_rng()
    chs_points: List[CHSPoint] = []
    chs_points.append(
        chs_point_normal(
            coords_mean=(0.0, 0.0, 0.0),
            coords_sigma=(0.0, 0.0, 0.0),
            direction_mean=(0, 0.6, 1.0),
            direction_sigma=(0.7, 0.2, 0.3),
            direction_magnitude_mean_and_sigma=(1.0, 0.1),
            diameter_mean_and_sigma=(22.0, 1.0),
            d_diameter_mean_and_sigma=(0.0, 0.0),
            coord_offset=start,
            rng=rng,
        )
    )
    chs_points.append(
        chs_point_normal(
            coords_mean=(5.0, 5.0, 25.0),
            coords_sigma=(1.5, 1.5, 3.0),
            direction_mean=(-0.3, 0.3, 0.7),
            direction_sigma=(0.3, 0.2, 0.3),
            direction_magnitude_mean_and_sigma=(0.8, 0.1),
            diameter_mean_and_sigma=(chs_points[-1].diameter - 10, 0.7),
            d_diameter_mean_and_sigma=(-0.1, 0.033),
            coord_offset=start,
            rng=rng,
        )
    )
    cl_coordinates, cl_radii = chs_to_cl_points(chs_points, resolution)
    return Branch("co", cl_coordinates, cl_radii), chs_points


def common_origin_IV(
    start: Tuple[float, float, float],
    resolution: float,
    rng: np.random.Generator = None,
) -> Tuple[Branch, List[CHSPoint]]:
    rng = rng or np.random.default_rng()
    chs_points: List[CHSPoint] = []
    chs_points.append(
        chs_point_normal(
            coords_mean=(0.0, 0.0, 0.0),
            coords_sigma=(0.0, 0.0, 0.0),
            direction_mean=(0, 0.1, 1),
            direction_sigma=(0.7, 0.6, 0.3),
            direction_magnitude_mean_and_sigma=(1.0, 0.1),
            diameter_mean_and_sigma=(22.0, 1.0),
            d_diameter_mean_and_sigma=(0.0, 0.0),
            coord_offset=start,
            rng=rng,
        )
    )
    chs_points.append(
        chs_point_normal(
            coords_mean=(-20.0, 5.0, 30.0),
            coords_sigma=(3.0, 2.0, 3.0),
            direction_mean=(-0.3, 0.3, 0.7),
            direction_sigma=(0.3, 0.2, 0.3),
            direction_magnitude_mean_and_sigma=(0.8, 0.1),
            diameter_mean_and_sigma=(chs_points[-1].diameter - 10, 0.7),
            d_diameter_mean_and_sigma=(-0.1, 0.033),
            coord_offset=start,
            rng=rng,
        )
    )
    cl_coordinates, cl_radii = chs_to_cl_points(chs_points, resolution)
    return Branch("co", cl_coordinates, cl_radii), chs_points
