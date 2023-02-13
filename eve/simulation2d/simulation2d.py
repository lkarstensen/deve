from abc import ABC, abstractmethod
import math
from typing import List, NamedTuple, Tuple

import numpy as np
import pymunk

from ..intervention import Intervention
from ..vesseltree import VesselTree
from .device import Device

from ..util.binaryimage import create_walls_from_vessel_tree, Wall
from ..util.insertionfromcenterline import calc_insertion_from_centerline
import logging


class Simulation2D(Intervention, ABC):
    def __init__(
        self,
        vessel_tree: VesselTree,
        devices: List[Device],
        velocity_limits: List[Tuple[float, float]],
        dimension_to_omit: str,
        image_frequency: float,
        dt_simulation: float,
        damping: float,
        last_segment_kp_angle: float,
        last_segment_kp_translation: float,
        tracking_low_custom: Tuple[float, float, float] = None,
        tracking_high_custom: Tuple[float, float, float] = None,
    ) -> None:
        super().__init__(vessel_tree, image_frequency, dt_simulation, velocity_limits)

        self.devices = devices
        self.damping = damping
        self.dimension_to_omit = dimension_to_omit
        self.last_segment_kp_angle = last_segment_kp_angle
        self.last_segment_kp_translation = last_segment_kp_translation
        self._space = None
        self._last_centerline_tree = None
        self._segment_bodies: List[pymunk.Body] = []
        self._walls: List[Tuple[pymunk.Body, pymunk.Shape]] = []
        self.logger = logging.getLogger(self.__module__)

    @abstractmethod
    def _remove_segments_and_constraints(self):
        ...

    @property
    def tracking_ground_truth(self):
        body_0_to_tip = self._segment_shapes[0].b.rotated(self._segment_bodies[0].angle)
        tip = self._segment_bodies[0].position + body_0_to_tip
        position = [tip]
        for body in self._segment_bodies:
            position.append(tuple(body.position))
        position = np.array(position)
        if self.dimension_to_omit == "y":
            insert_idx = 1
        elif self.dimension_to_omit == "z":
            insert_idx = 2
        else:
            insert_idx = 0
        position = np.insert(position, insert_idx, 0, axis=1)
        return position

    def reset(self, episode_nr=0) -> None:
        if self._last_centerline_tree != self.vessel_tree.branches:
            if self._space is not None:
                self._remove_segments_and_constraints()
                self._remove_walls()
                del self._space
                self._space = None
            self._create_space()
            walls = create_walls_from_vessel_tree(
                self.vessel_tree,
                self.dimension_to_omit,
                pixel_spacing=0.05,
                contour_approx_margin=6.0,
            )
            for wall in walls:
                self._add_wall(wall)
            self._last_centerline_tree = self.vessel_tree.branches

            ip_pos, ip_dir = calc_insertion_from_centerline(self.vessel_tree)
            if self.dimension_to_omit == "x":
                ip_pos = np.array([ip_pos[1], ip_pos[2]])
                ip_dir = np.array([ip_dir[1], ip_dir[2]])
            elif self.dimension_to_omit == "z":
                ip_pos = np.array([ip_pos[0], ip_pos[1]])
                ip_dir = np.array([ip_dir[0], ip_dir[1]])
            else:
                ip_pos = np.array([ip_pos[0], ip_pos[2]])
                ip_dir = np.array([ip_dir[0], ip_dir[2]])
            self.insertion_point = ip_pos
            self.insertion_direction = ip_dir

            self.reset_devices()

        self._last_action = np.zeros_like(self.velocity_limits)

    def close(self) -> None:
        self._remove_segments_and_constraints()
        self._remove_walls()
        del self._space
        self._space = None

    def _create_space(self):
        self._space = pymunk.Space()
        self.static_body = self._space.static_body
        self._space.gravity = 0.0, 0.0
        self._space.damping = self.damping

    def _add_wall(self, wall: Wall):
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        shape = pymunk.Segment(body, wall.start.tolist(), wall.end.tolist(), 0.0)
        shape.friction = 0.0
        shape.elasticity = 0.0
        self._space.add(body)
        self._space.add(shape)
        self._walls.append([body, shape])

    def _remove_walls(self):
        for wall in self._walls:
            self._space.remove(*wall)
        del self._walls
        self._walls = []

    def _set_insertion_point_stabilisation(self):
        ip = pymunk.Vec2d(*self.insertion_point)
        ip_dir = pymunk.Vec2d(*self.insertion_direction)
        last_segment_to_ip = self._segment_bodies[-1].position - ip
        last_segment_perpendicular_dot_product = last_segment_to_ip.dot(
            ip_dir.rotated(math.pi / 2)
        )
        self._segment_bodies[-1].velocity = (
            -ip_dir.rotated(math.pi / 2)
            * last_segment_perpendicular_dot_product
            * self.last_segment_kp_translation
        )

        last_segment_angle = self._segment_bodies[-1].angle
        # correcting angular velocity to keep insertion direction
        self._segment_bodies[-1].angular_velocity = (
            -last_segment_angle * self.last_segment_kp_angle
        )

    def _set_base_translation(self, trans_velocity):
        ip_dir = pymunk.Vec2d(*self.insertion_direction)
        self._segment_bodies[-1].position += (
            ip_dir * trans_velocity * self.dt_simulation
        )
