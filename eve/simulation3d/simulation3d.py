from abc import ABC, abstractmethod
import numpy as np

from ..intervention import Intervention, List, Tuple
from ..vesseltree import VesselTree


class Simulation3D(Intervention, ABC):
    def __init__(
        self,
        vessel_tree: VesselTree,
        image_frequency: float,
        dt_simulation: float,
        velocity_limits: List[Tuple[float, float]],
    ) -> None:
        super().__init__(vessel_tree, image_frequency, dt_simulation, velocity_limits)
        self.initialized_last_reset = True
        self.root = None
        self.sofa_initialized_2 = False

    @abstractmethod
    def _unload_sofa(
        self,
    ):
        ...

    @abstractmethod
    def _do_sofa_step(self, action: np.ndarray):
        ...

    @abstractmethod
    def _reset_sofa_devices(self):
        ...

    @abstractmethod
    def _init_sofa(
        self,
        insertion_point: np.ndarray,
        insertion_direction: np.ndarray,
        mesh_path: str,
    ):
        ...

    @staticmethod
    def _calculate_insertion_pose(
        insertion_point: np.ndarray, insertion_direction: np.ndarray
    ):

        insertion_direction = insertion_direction / np.linalg.norm(insertion_direction)
        original_direction = np.array([1.0, 0.0, 0.0])
        if np.all(insertion_direction == original_direction):
            w0 = 1.0
            xyz0 = [0.0, 0.0, 0.0]
        elif np.all(np.cross(insertion_direction, original_direction) == 0):
            w0 = 0.0
            xyz0 = [0.0, 1.0, 0.0]
        else:
            half = (original_direction + insertion_direction) / np.linalg.norm(
                original_direction + insertion_direction
            )
            w0 = np.dot(original_direction, half)
            xyz0 = np.cross(original_direction, half)
        xyz0 = list(xyz0)
        pose = list(insertion_point) + list(xyz0) + [w0]
        return pose

    def _calc_euler_angles(self, vector):
        a = np.array([1, 0, 0])
        a = a / np.linalg.norm(a)
        b = vector
        b = b / np.linalg.norm / b
        v = np.cross(a, b)
        vx = [
            [0, -v[3], v[2]],
            [v[3], 0, -v[1]],
            [-v[2], v[1], 0],
        ]
        c = np.dot(a, b)
        I = np.eye(3)
        R = I + vx + vx**2 * (1 / (1 + c))
        R = round(R, 5)
        cos = np.cos
        atan2 = np.arctan2
        if R[3, 1] != 1 and R[3, 1] != -1:
            theta_1 = -np.arcsin(R[3, 1])
            theta_2 = np.pi - theta_1
            chi_1 = np.arctan2((R[3, 2] / cos(theta_1)), (R[3, 3] / cos(theta_1)))
            chi_2 = atan2((R[3, 2] / cos(theta_2)), (R[3, 3] / cos(theta_2)))
            phi_1 = atan2((R[2, 1] / cos(theta_1)), (R[1, 1] / cos(theta_1)))
            phi_2 = atan2((R[2, 1] / cos(theta_2)), (R[1, 1] / cos(theta_2)))
            theta = min(theta_1, theta_2)
            chi = min(chi_1, chi_2)
            phi = min(phi_1, phi_2)
        else:
            phi = 0
            if R[3, 1] == -1:
                theta = np.pi / 2
                chi = phi + atan2(R[1, 2], R[1, 3])
            else:
                theta = -np.pi / 2
                chi = -phi + atan2(-R[1, 2], -R[1, 3])

        theta = np.rad2deg(theta)
        chi = np.rad2deg(chi)
        phi = np.rad2deg(phi)
        return theta, chi, phi
