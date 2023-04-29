from typing import List, Tuple

from ..branch import Branch
from ..util.cubichermitesplines import CHSPoint, chs_point_normal, chs_to_cl_points

import numpy as np


def left_common_carotid(
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
            direction_mean=(0.3, 0.0, 1.0),
            direction_sigma=(0.3, 0.3, 0.0),
            direction_magnitude_mean_and_sigma=(1.5, 0.2),
            diameter_mean_and_sigma=(10.0, 0.33),
            d_diameter_mean_and_sigma=(0.0, 0.0),
            coord_offset=start,
            rng=rng,
        )
    )
    chs_points.append(
        chs_point_normal(
            coords_mean=(2.0, 8.0, 230 - start[2]),
            coords_sigma=(1.0, 3.0, 0.0),
            direction_mean=(0.0, 0.0, 1.0),
            direction_sigma=(0.2, 0.3, 0.0),
            direction_magnitude_mean_and_sigma=(1.5, 0.2),
            diameter_mean_and_sigma=(chs_points[-1].diameter - 0.5, 0.2),
            d_diameter_mean_and_sigma=(0.0, 0.0),
            coord_offset=start,
            rng=rng,
        )
    )
    cl_coordinates, cl_radii = chs_to_cl_points(chs_points, resolution)
    return Branch("lcca", cl_coordinates, cl_radii), chs_points


def left_common_carotid_II(
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
            direction_mean=(0.6, 0.0, 1.0),
            direction_sigma=(0.2, 0.3, 0.3),
            direction_magnitude_mean_and_sigma=(1.5, 0.2),
            diameter_mean_and_sigma=(10.0, 0.33),
            d_diameter_mean_and_sigma=(0.0, 0.0),
            coord_offset=start,
            rng=rng,
        )
    )
    chs_points.append(
        chs_point_normal(
            coords_mean=(25.0, 8.0, 230 - start[2]),
            coords_sigma=(1.0, 3.0, 0.0),
            direction_mean=(0.0, 0.0, 1.0),
            direction_sigma=(0.2, 0.3, 0.0),
            direction_magnitude_mean_and_sigma=(1.5, 0.2),
            diameter_mean_and_sigma=(chs_points[-1].diameter - 0.5, 0.2),
            d_diameter_mean_and_sigma=(0.0, 0.0),
            coord_offset=start,
            rng=rng,
        )
    )
    cl_coordinates, cl_radii = chs_to_cl_points(chs_points, resolution)
    return Branch("lcca", cl_coordinates, cl_radii), chs_points
