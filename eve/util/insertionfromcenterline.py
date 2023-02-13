from typing import Tuple
from ..vesseltree import VesselTree
import numpy as np


def calc_insertion_from_centerline(
    centerline_tree: VesselTree,
) -> Tuple[np.ndarray, np.ndarray]:
    point_0 = centerline_tree["aorta"].coordinates[0]
    point_1 = centerline_tree["aorta"].coordinates[1]
    insertion_point = point_0
    insertion_direction = point_1 - point_0
    insertion_direction = insertion_direction / np.linalg.norm(insertion_direction)
    return insertion_point, insertion_direction
