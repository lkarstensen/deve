from typing import List, Tuple

from ..util.branch import BranchWithRadii
from ..util.cubichermitesplines import CHSPoint, chs_point_normal, chs_to_cl_points

import numpy as np


def left_subclavian(
    start: Tuple[float, float, float],
    resolution: float,
    rng: np.random.Generator = None,
) -> Tuple[BranchWithRadii, List[CHSPoint]]:
    rng = rng or np.random.default_rng()

    chs_points: List[CHSPoint] = []
    chs_points.append(
        chs_point_normal(
            coords_mean=(0.0, 0.0, 0.0),
            coords_sigma=(0.0, 0.0, 0.0),
            direction_mean=(0.1, 0.1, 2.0),
            direction_sigma=(0.1, 0.1, 0.3),
            direction_magnitude_mean_and_sigma=(2.0, 0.4),
            radius_mean_and_sigma=(10.0, 0.33),
            d_radius_mean_and_sigma=(0.0, 0.0),
            coord_offset=start,
            rng=rng,
        )
    )
    chs_points.append(
        chs_point_normal(
            coords_mean=(59.0 - start[0], 5.0, 50.0),
            coords_sigma=(0.0, 2.0, 2.0),
            direction_mean=(1.0, 0.3, 0.0),
            direction_sigma=(0.2, 0.07, 0.15),
            direction_magnitude_mean_and_sigma=(1.7, 0.4),
            radius_mean_and_sigma=(chs_points[-1].r - 1, 0.33),
            d_radius_mean_and_sigma=(-0.1, 0.03),
            coord_offset=start,
            rng=rng,
        )
    )
    cl_coordinates, cl_radii = chs_to_cl_points(chs_points, resolution)
    return BranchWithRadii("lsa", cl_coordinates, cl_radii), chs_points


def left_subclavian_IV(
    start: Tuple[float, float, float],
    resolution: float,
    rng: np.random.Generator = None,
) -> Tuple[BranchWithRadii, List[CHSPoint]]:
    rng = rng or np.random.default_rng()

    chs_points: List[CHSPoint] = []
    chs_points.append(
        chs_point_normal(
            coords_mean=(0.0, 0.0, 0.0),
            coords_sigma=(0.0, 0.0, 0.0),
            direction_mean=(0.1, 0.1, 2.0),
            direction_sigma=(0.1, 0.1, 0.3),
            direction_magnitude_mean_and_sigma=(2.0, 0.4),
            radius_mean_and_sigma=(10.0, 0.33),
            d_radius_mean_and_sigma=(0.0, 0.0),
            coord_offset=start,
            rng=rng,
        )
    )
    chs_points.append(
        chs_point_normal(
            # x is fixed
            coords_mean=(59.0 - start[0], 5.0, 40.0),
            coords_sigma=(0.0, 2.0, 2.0),
            direction_mean=(1.0, 0.3, 0.0),
            direction_sigma=(0.2, 0.07, 0.15),
            direction_magnitude_mean_and_sigma=(1.7, 0.4),
            radius_mean_and_sigma=(chs_points[-1].r - 1, 0.33),
            d_radius_mean_and_sigma=(-0.1, 0.03),
            coord_offset=start,
            rng=rng,
        )
    )
    cl_coordinates, cl_radii = chs_to_cl_points(chs_points, resolution)
    return BranchWithRadii("lsa", cl_coordinates, cl_radii), chs_points


def left_subclavian_VI(
    start: Tuple[float, float, float],
    resolution: float,
    rng: np.random.Generator = None,
) -> Tuple[BranchWithRadii, List[CHSPoint]]:
    rng = rng or np.random.default_rng()

    chs_points: List[CHSPoint] = []
    chs_points.append(
        chs_point_normal(
            coords_mean=(0.0, 0.0, 0.0),
            coords_sigma=(0.0, 0.0, 0.0),
            direction_mean=(0.1, 0.1, 2.0),
            direction_sigma=(0.1, 0.1, 0.3),
            direction_magnitude_mean_and_sigma=(2.0, 0.4),
            radius_mean_and_sigma=(10.0, 0.33),
            d_radius_mean_and_sigma=(0.0, 0.0),
            coord_offset=start,
            rng=rng,
        )
    )
    chs_points.append(
        chs_point_normal(
            # x is fixed
            coords_mean=(59.0 - start[0], 15.0, 20.0),
            coords_sigma=(0.0, 2.0, 2.0),
            direction_mean=(1.0, 0.3, -0.1),
            direction_sigma=(0.2, 0.07, 0.15),
            direction_magnitude_mean_and_sigma=(1.7, 0.4),
            radius_mean_and_sigma=(chs_points[-1].r - 1, 0.33),
            d_radius_mean_and_sigma=(-0.1, 0.03),
            coord_offset=start,
            rng=rng,
        )
    )
    cl_coordinates, cl_radii = chs_to_cl_points(chs_points, resolution)
    return BranchWithRadii("lsa", cl_coordinates, cl_radii), chs_points
