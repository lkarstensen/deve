import math
import pymunk
from typing import List, Tuple
import numpy as np

from .simulation2d import Simulation2D
from ..vesseltree import VesselTree
from ..pathfinder import Pathfinder
from .device import Device
import logging


def clamp(x, minimum, maximum):
    return max(minimum, min(x, maximum))


class SingleDevice(Simulation2D):
    def __init__(
        self,
        vessel_tree: VesselTree,
        device: Device,
        velocity_limit: Tuple[float, float] = (50, 3.14),
        element_length: float = 3,
        dimension_to_omit: str = "y",
        stop_device_at_tree_end: bool = True,
        image_frequency: float = 7.5,
        dt_simulation: float = 2.5 / 10000,
        friction: float = 1.0,
        damping: float = 0.000001,
        body_mass: float = 0.01,
        body_moment: float = 0.1,
        linear_stiffness: float = 2.5e6,
        linear_damping: float = 100,
        last_segment_kp_angle: float = 3,
        last_segment_kp_translation: float = 5,
        tracking_low_custom: Tuple[float, float, float] = None,
        tracking_high_custom: Tuple[float, float, float] = None,
    ) -> None:
        super().__init__(
            vessel_tree=vessel_tree,
            devices=[device],
            velocity_limits=[velocity_limit],
            dimension_to_omit=dimension_to_omit,
            image_frequency=image_frequency,
            dt_simulation=dt_simulation,
            damping=damping,
            last_segment_kp_angle=last_segment_kp_angle,
            last_segment_kp_translation=last_segment_kp_translation,
            tracking_low_custom=tracking_low_custom,
            tracking_high_custom=tracking_high_custom,
        )
        self.device = device
        self.element_length = element_length
        self.friction = friction
        self.velocity_limit = np.array(velocity_limit)
        self.stop_device_at_tree_end = stop_device_at_tree_end

        self.body_mass = body_mass
        self.body_moment = body_moment
        self.linear_stiffness = linear_stiffness
        self.linear_damping = linear_damping

        self._segment_shapes: List[pymunk.shapes.Shape] = []
        self._joints_and_springs: List[Tuple[pymunk.Constraint, pymunk.Constraint]] = []
        self._centerline_tree = None
        self.device.element_length = element_length
        self.logger = logging.getLogger(self.__module__)

    @property
    def device_rotations(self) -> Tuple[float]:
        return [self.device.rotation]

    @property
    def tracking_per_device(self) -> List[np.ndarray]:
        return [self.tracking]

    @property
    def device_lengths_inserted(self) -> List[float]:
        return [self.device.inserted_length]

    @property
    def device_lengths_maximum(self) -> List[float]:
        return [self.device.total_length]

    @property
    def device_diameters(self) -> List[float]:
        return [self.device.diameter]

    def step(self, action: np.ndarray) -> None:
        action = np.array(action)
        action = action.reshape(self.action_space.shape)
        action = np.clip(action, -self.velocity_limits, self.velocity_limits)
        self._last_action = action

        self.device.trans_velocity = action[0, 0]
        self.device.rot_velocity = action[0, 1]

        tip = self.tracking[0]
        if (
            self.stop_device_at_tree_end
            and self.vessel_tree.at_tree_end(tip)
            and self.device.trans_velocity > 0
            and self.device.inserted_length > 10
        ):
            self.device.trans_velocity = 0.0

        for _ in range(int((1 / self.image_frequency) / self.dt_simulation)):
            self.device.simu_step(self.dt_simulation)
            self._add_or_remove_wire_element(self.device.trans_velocity)
            self._set_rotation()
            self._set_base_translation(self.device.trans_velocity)
            self._set_insertion_point_stabilisation()

            self._space.step(self.dt_simulation)

    def _remove_segments_and_constraints(self):
        for joint_spring in self._joints_and_springs:
            self._space.remove(*joint_spring)
        self._space.remove(*self._segment_shapes)
        self._space.remove(*self._segment_bodies)
        del self._joints_and_springs
        del self._segment_bodies
        del self._segment_shapes
        self._segment_bodies, self._segment_shapes, self._joints_and_springs = (
            [],
            [],
            [],
        )

    def reset_devices(self):
        self._remove_segments_and_constraints()
        self.device.reset()
        self._add_element()

    def _add_or_remove_wire_element(self, trans_velocity):
        n_elements = len(self._segment_bodies)
        d_n_elements = int(self.device.inserted_length / self.element_length) + 1
        if d_n_elements > n_elements:
            self._add_element()

        elif d_n_elements < n_elements:
            self._remove_element()

    def _add_element(self):
        ip = pymunk.Vec2d(*self.insertion_point)
        ip_dir = pymunk.Vec2d(*self.insertion_direction)
        nth_element = len(self._segment_bodies)
        segment_position = self._segment_bodies[-1].position if nth_element > 0 else ip
        segment_angle = self._segment_bodies[-1].angle if nth_element > 0 else 0.0
        nth_element = len(self._segment_bodies)

        wire_segment_body = pymunk.Body(mass=self.body_mass, moment=self.body_moment)
        wire_segment_body.position = segment_position - ip_dir * self.element_length

        wire_segment_body.angle = segment_angle
        wire_segment_shape = pymunk.shapes.Segment(
            wire_segment_body,
            pymunk.Vec2d(0, 0),
            ip_dir * self.element_length,
            self.device.diameter / 2,
        )
        wire_segment_shape.filter = pymunk.ShapeFilter(group=1)

        self._space.add(wire_segment_body, wire_segment_shape)
        self._segment_bodies.append(wire_segment_body)
        self._segment_shapes.append(wire_segment_shape)

        if nth_element > 0:

            joint = pymunk.DampedSpring(
                self._segment_bodies[-2],
                wire_segment_body,
                (0, 0),
                self._segment_bodies[-2].position - wire_segment_body.position,
                0.0,
                self.linear_stiffness,
                self.linear_damping,
            )
            spring = pymunk.DampedRotarySpring(
                self._segment_bodies[-2],
                wire_segment_body,
                0.0,
                self.device.get_last_inserted_element_stiffness(),
                self.device.get_last_inserted_element_damping(),
            )
            spring.collide_bodies = False
            self._space.add(joint, spring)
            self._joints_and_springs.append([joint, spring])

    def _remove_element(self):
        if self._joints_and_springs:
            joint_spring = self._joints_and_springs.pop(-1)
            self._space.remove(*joint_spring)
            body = self._segment_bodies.pop(-1)
            shape = self._segment_shapes.pop(-1)
            self._space.remove(body, shape)

    def _set_rotation(self):
        rest_angles, _ = self.device.get_rest_angles_with_stiffnesses()
        for i in range(rest_angles.shape[0]):
            if i < len(self._joints_and_springs):
                self._joints_and_springs[i][1].rest_angle = rest_angles[i]
            else:
                break
