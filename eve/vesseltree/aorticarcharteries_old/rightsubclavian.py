from typing import List, Tuple

from ..branch import Branch
from ..util.cubichermitesplines import CHSPoint, chs_point_normal, chs_to_cl_points

import numpy as np


def right_subclavian(
    start: Tuple[float, float, float],
    start_cp_chs: CHSPoint,
    resolution: float,
    rng: np.random.Generator = None,
) -> Tuple[Branch, List[CHSPoint]]:
    rng = rng or np.random.default_rng()
    chs_points: List[CHSPoint] = []
    chs_points.append(
        chs_point_normal(
            coords_mean=(0.0, 0.0, 0.0),
            coords_sigma=(0.0, 0.0, 0.0),
            direction_mean=(
                start_cp_chs.d_coords[0],
                start_cp_chs.d_coords[1],
                start_cp_chs.d_coords[2],
            ),
            direction_sigma=(0.0, 0.0, 0.0),
            direction_magnitude_mean_and_sigma=(1.5, 0.2),
            diameter_mean_and_sigma=(start_cp_chs.diameter, 0.0),
            d_diameter_mean_and_sigma=(start_cp_chs.d_diameter, 0.0),
            coord_offset=start,
            rng=rng,
        )
    )

    chs_points.append(
        chs_point_normal(
            coords_mean=(-57.0 - start[0], 35.0, 25.0),
            coords_sigma=(0.0, 2.0, 1.0),
            direction_mean=(-1.2, 0.0, 0.0),
            direction_sigma=(0.3, 0.2, 0.1),
            direction_magnitude_mean_and_sigma=(1.5, 0.2),
            diameter_mean_and_sigma=(chs_points[-1].diameter, 0.0),
            d_diameter_mean_and_sigma=(0.0, 0.0),
            coord_offset=start,
            rng=rng,
        )
    )
    cl_coordinates, cl_radii = chs_to_cl_points(chs_points, resolution)
    return Branch("rsa", cl_coordinates, cl_radii), chs_points


def right_subclavian_IV(
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
            direction_mean=(0.1, 0.1, 2.0),
            direction_sigma=(0.1, 0.1, 0.3),
            direction_magnitude_mean_and_sigma=(2.0, 0.4),
            diameter_mean_and_sigma=(10.0, 0.33),
            d_diameter_mean_and_sigma=(0.0, 0.0),
            coord_offset=start,
            rng=rng,
        )
    )
    chs_points.append(
        chs_point_normal(
            coords_mean=(-57.0 - start[0], 2.0, 50.0),
            coords_sigma=(0.0, 2.0, 1.0),
            direction_mean=(-1.2, 0.0, 0.0),
            direction_sigma=(0.3, 0.2, 0.1),
            direction_magnitude_mean_and_sigma=(1.5, 0.2),
            diameter_mean_and_sigma=(chs_points[-1].diameter, 0.0),
            d_diameter_mean_and_sigma=(0.0, 0.0),
            coord_offset=start,
            rng=rng,
        )
    )
    cl_coordinates, cl_radii = chs_to_cl_points(chs_points, resolution)
    return Branch("rsa", cl_coordinates, cl_radii), chs_points


def right_subclavian_V(
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
            direction_mean=(0.1, 0.1, 2.0),
            direction_sigma=(0.1, 0.1, 0.3),
            direction_magnitude_mean_and_sigma=(2.0, 0.4),
            diameter_mean_and_sigma=(10.0, 0.33),
            d_diameter_mean_and_sigma=(0.0, 0.0),
            coord_offset=start,
            rng=rng,
        )
    )
    chs_points.append(
        chs_point_normal(
            coords_mean=(-57.0 - start[0], 2.0, 42.0),
            coords_sigma=(0.0, 2.0, 1.0),
            direction_mean=(-1.2, -2.5, 0.0),
            direction_sigma=(0.3, 0.2, 0.1),
            direction_magnitude_mean_and_sigma=(1.5, 0.2),
            diameter_mean_and_sigma=(chs_points[-1].diameter, 0.0),
            d_diameter_mean_and_sigma=(0.0, 0.0),
            coord_offset=start,
            rng=rng,
        )
    )
    cl_coordinates, cl_radii = chs_to_cl_points(chs_points, resolution)
    return Branch("rsa", cl_coordinates, cl_radii), chs_points


def right_subclavian_VI(
    start: Tuple[float, float, float],
    start_cp_chs: CHSPoint,
    resolution: float,
    rng: np.random.Generator = None,
) -> Tuple[Branch, List[CHSPoint]]:
    rng = rng or np.random.default_rng()
    chs_points: List[CHSPoint] = []
    chs_points.append(
        chs_point_normal(
            coords_mean=(0.0, 0.0, 0.0),
            coords_sigma=(0.0, 0.0, 0.0),
            direction_mean=(
                start_cp_chs.d_coords[0],
                start_cp_chs.d_coords[1],
                start_cp_chs.d_coords[2],
            ),
            direction_sigma=(0.0, 0.0, 0.0),
            direction_magnitude_mean_and_sigma=(1.5, 0.2),
            diameter_mean_and_sigma=(start_cp_chs.diameter, 0.0),
            d_diameter_mean_and_sigma=(start_cp_chs.d_diameter, 0.0),
            coord_offset=start,
            rng=rng,
        )
    )

    chs_points.append(
        chs_point_normal(
            coords_mean=(-57.0 - start[0], 15.0, 20.0),
            coords_sigma=(0.0, 2.0, 1.0),
            direction_mean=(-0.6, 0.0, -0.1),
            direction_sigma=(0.1, 0.2, 0.1),
            direction_magnitude_mean_and_sigma=(1.5, 0.2),
            diameter_mean_and_sigma=(chs_points[-1].diameter, 0.0),
            d_diameter_mean_and_sigma=(0.0, 0.0),
            coord_offset=start,
            rng=rng,
        )
    )
    cl_coordinates, cl_radii = chs_to_cl_points(chs_points, resolution)
    return Branch("rsa", cl_coordinates, cl_radii), chs_points
