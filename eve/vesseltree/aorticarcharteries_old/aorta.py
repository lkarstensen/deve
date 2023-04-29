from typing import List, Tuple

from ..branch import Branch
from ..util.cubichermitesplines import CHSPoint, chs_point_normal, chs_to_cl_points

import numpy as np


def aorta_generator(
    resolution: float,
    rng: np.random.Generator = None,
) -> Tuple[Branch, List[CHSPoint]]:
    rng = rng or np.random.default_rng()
    chs_points: List[CHSPoint] = []
    chs_points.append(
        chs_point_normal(
            coords_mean=(0.0, 0.0, 0.0),
            coords_sigma=(0.0, 0.0, 0.0),
            direction_mean=(0, 0.0, 1),
            direction_sigma=(0.1, 0.00, 0.3),
            direction_magnitude_mean_and_sigma=(1.0, 0.1),
            diameter_mean_and_sigma=(20.0, 1.0),
            d_diameter_mean_and_sigma=(0.02, 0.005),
            rng=rng,
        )
    )
    chs_points.append(
        chs_point_normal(
            coords_mean=(
                30.0,
                0.0,
                90.0,
            ),
            coords_sigma=(2.0, 1.0, 3.0),
            direction_mean=(1, 0, 2),
            direction_sigma=(0.4, 0.3, 0.2),
            direction_magnitude_mean_and_sigma=(1.2, 0.15),
            diameter_mean_and_sigma=(chs_points[-1].diameter + 5, 0.7),
            d_diameter_mean_and_sigma=(0.02, 0.005),
            rng=rng,
        )
    )
    chs_points.append(
        chs_point_normal(
            coords_mean=(20.0, -40.0, 160.0),
            coords_sigma=(1.5, 3.0, 2.0),
            direction_mean=(-4, -6, -0.5),
            direction_sigma=(0.8, 1, 0.7),
            direction_magnitude_mean_and_sigma=(1.4, 0.15),
            diameter_mean_and_sigma=(chs_points[-1].diameter + 7, 0.7),
            d_diameter_mean_and_sigma=(0.07, 0.008),
            rng=rng,
        )
    )

    chs_points.append(
        chs_point_normal(
            coords_mean=(-5.0, -60.0, 100.0),
            coords_sigma=(1.0, 2.0, 1.0),
            direction_mean=(3, 4, -8),
            direction_sigma=(0.3, 0.3, 0.5),
            direction_magnitude_mean_and_sigma=(1.6, 0.15),
            diameter_mean_and_sigma=(chs_points[-1].diameter + 8, 0.7),
            d_diameter_mean_and_sigma=(0.0, 0.0),
            rng=rng,
        )
    )
    cl_coordinates, cl_radii = chs_to_cl_points(chs_points, resolution)
    return Branch("aorta", cl_coordinates, cl_radii), chs_points
