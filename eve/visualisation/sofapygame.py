from math import cos, sin
from typing import Tuple
from OpenGL.GL import (
    glClear,
    glEnable,
    glMatrixMode,
    glLoadIdentity,
    glMultMatrixd,
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_BUFFER_BIT,
    GL_LIGHTING,
    GL_DEPTH_TEST,
    GL_PROJECTION,
    GL_MODELVIEW,
)
from OpenGL.GLU import gluPerspective
import pygame
import Sofa
import Sofa.SofaGL
import numpy as np

from .visualisation import Visualisation
from ..simulation3d.simulation3d import Simulation3D
from ..interimtarget import InterimTarget, InterimTargetDummy
from ..target import Target as TargetClass


class SofaPygame(Visualisation):
    def __init__(
        self,
        intervention: Simulation3D,
        display_size: Tuple[float, float] = (1200, 860),
        interim_target: InterimTarget = None,
        target: TargetClass = None,
    ) -> None:
        self.intervention = intervention
        self.display_size = display_size
        self.interim_target = interim_target or InterimTargetDummy()
        self.target = target

        self.intervention.init_visual = True

        self.initial_orientation = None
        self._initialized = False
        self._centerline_tree = None
        self._interim_target_spheres = {}
        self._main_target = None
        self._interim_targets_node = None
        self._camera = None
        self._theta_x = None
        self._theta_z = None
        self._initial_direction = None
        self._distance = None
        self._field_of_view = None
        self._z_near = None
        self._z_far = None

    def step(self) -> None:

        visu_set = set(list(self._interim_target_spheres.keys()))
        target_set = set(tuple(map(tuple, self.interim_target.all_coordinates)))
        missing_targets = visu_set - target_set
        for target in missing_targets:
            sphere = self._interim_target_spheres.pop(target)
            sphere.removeObject(sphere.OglModel)
            self._interim_targets_node.removeChild(sphere)

        if self.initial_orientation is None:

            self.initial_orientation = np.array(
                [
                    self._camera.orientation[3],
                    self._camera.orientation[0],
                    self._camera.orientation[1],
                    self._camera.orientation[2],
                ]
            )

        Sofa.Simulation.updateVisual(self.intervention.root)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_LIGHTING)
        glEnable(GL_DEPTH_TEST)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(
            self._field_of_view,
            (self.display_size[0] / self.display_size[1]),
            self._z_near,
            self._z_far,
        )
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        camera_mvm = self._camera.getOpenGLModelViewMatrix()
        glMultMatrixd(camera_mvm)
        Sofa.SofaGL.draw(self.intervention.root)

        pygame.display.get_surface().fill((0, 0, 0))
        pygame.display.flip()
        pygame.event.get()

    def reset(self, episode_nr: int = 0) -> None:

        targets = list(self._interim_target_spheres.keys())
        for target in targets:
            sphere = self._interim_target_spheres.pop(target)
            self._interim_targets_node.removeChild(sphere)
        if self.intervention.initialized_last_reset:
            self._main_target = None
        if self._main_target is not None:
            target_coord = self.target.coordinates.tolist()
            self._main_target.ball.translation = target_coord
            Sofa.Simulation.init(self._main_target)

        if not self._initialized or self.intervention.initialized_last_reset:
            self._interim_target_spheres = {}
            self.intervention.root.addObject("RequiredPlugin", name="SofaGeneralLoader")
            self._interim_targets_node = self.intervention.root.addChild(
                "InterimTargets"
            )
            self._interim_targets_node.addObject(
                "MeshSTLLoader",
                name="loader",
                triangulate=True,
                filename="/Users/lennartkarstensen/stacie/eve/eve/visualisation/meshes/unit_sphere.stl",
                scale=self.interim_target.threshold,
            )
            self._interim_targets_node.addObject("MechanicalObject", src="@loader")

        for i, interim_target in enumerate(self.interim_target.all_coordinates):
            interim_target = tuple(interim_target)
            target_node = self._interim_targets_node.addChild("target_" + str(i))

            target_node.addObject(
                "OglModel",
                src="@../loader",
                color=[0.5, 1.0, 1.0, 0.3],
                translation=interim_target,
            )
            self._interim_target_spheres[interim_target] = target_node

        if self.target is not None and self._main_target is None:
            target_coord = tuple(self.target.coordinates)

            target_node = self.intervention.root.addChild("main_target")
            target_node.addObject(
                "MeshSTLLoader",
                name="loader",
                triangulate=True,
                filename="/Users/lennartkarstensen/stacie/eve/eve/visualisation/meshes/unit_sphere.stl",
                scale=self.target.threshold,
            )
            target_node.addObject(
                "MechanicalObject",
                src="@loader",
                translation=(
                    target_coord[0],
                    target_coord[1],
                    target_coord[2] - self.target.threshold,
                ),
                template="Rigid3d",
                name="ball",
            )
            target_node.addObject(
                "OglModel",
                src="@loader",
                color=[0.5, 0.0, 0.0, 0.7],
                # translation=target_coord,
                name="ogl_model",
            )
            target_node.addObject("RigidMapping", input="@ball")
            self._main_target = target_node
        if not self._initialized or self.intervention.initialized_last_reset:
            self.intervention.root.addObject(
                "RequiredPlugin", name="SofaOpenglVisual"
            )  # visual stuff
            self.intervention.root.addObject(
                "RequiredPlugin", name="SofaGeneralTopology"
            )

            self.intervention.root.addObject("RequiredPlugin", name="SofaSimpleFem")
            self.intervention.root.addObject("DefaultVisualManagerLoop")
            self.intervention.root.addObject(
                "VisualStyle",
                displayFlags="showVisualModels showBehaviorModels hideCollisionModels hideMappings showForceFields",
            )
            self.intervention.root.addObject("LightManager")
            self.intervention.root.addObject("DirectionalLight", direction=[0, -1, 0])
            self.intervention.root.addObject("DirectionalLight", direction=[0, 1, 0])
            self.intervention.root.addObject("BackgroundSetting", color="0 0 0")

            look_at = (
                self.intervention.tracking_space.high
                + self.intervention.tracking_space.low
            ) * 0.5
            distance_coefficient = 1.5
            distance = (
                np.linalg.norm(look_at - self.intervention.tracking_space.low)
                * distance_coefficient
            )
            position = look_at + np.array([0.0, -distance, 0.0])

            scene_radius = np.linalg.norm(
                self.intervention.tracking_space.high
                - self.intervention.tracking_space.low
            )
            dist_cam_to_center = np.linalg.norm(position - look_at)

            z_clipping_coeff = 5
            z_near_coeff = 0.01

            z_near = dist_cam_to_center - scene_radius
            z_far = (z_near + 2 * scene_radius) * 2
            z_near = z_near * z_near_coeff

            z_min = z_near_coeff * z_clipping_coeff * scene_radius

            if z_near < z_min:
                z_near = z_min

            field_of_view = 70
            look_at = np.array(look_at)
            position = np.array(position)

            self._camera = self.intervention.root.addObject(
                "Camera",
                name="camera",
                lookAt=look_at,
                position=position,
                fieldOfView=field_of_view,
                zNear=z_near,
                zFar=z_far,
                fixedLookAt=False,
                # backgroundSetting="@BackgroundSetting",
            )
            self._theta_x = 0
            self._theta_z = 0
            self._initial_direction = position - look_at
            self._initial_direction = self._initial_direction / np.linalg.norm(
                self._initial_direction
            )
            self._distance = distance
            self._field_of_view = field_of_view
            self._z_near = z_near
            self._z_far = z_far
            pygame.display.init()
            pygame.display.set_mode(
                self.display_size,
                pygame.DOUBLEBUF | pygame.OPENGL,  # pylint: disable=no-member
            )

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glEnable(GL_LIGHTING)
            glEnable(GL_DEPTH_TEST)
            Sofa.SofaGL.glewInit()
            Sofa.Simulation.initVisual(self.intervention.root)
            Sofa.Simulation.initTextures(self.intervention.root)
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            gluPerspective(
                self._field_of_view,
                (self.display_size[0] / self.display_size[1]),
                self._z_near,
                self._z_far,
            )

            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            self._initialized = True
            if self.intervention.sofa_initialized_2:
                self.initial_orientation = np.array(
                    [
                        self._camera.orientation[3],
                        self._camera.orientation[0],
                        self._camera.orientation[1],
                        self._camera.orientation[2],
                    ]
                )
            else:
                self.initial_orientation = None

    def close(self):

        pygame.quit()  # pylint: disable=no-member

    def translate(self, velocity: np.array):

        dt = self.intervention.root.dt.value

        position = self._camera.position
        position += velocity * dt
        self._camera.position = position

        look_at = self._camera.lookAt
        look_at += velocity * dt
        self._camera.lookAt = look_at

    def zoom(self, velocity: float):
        position = self._camera.position
        look_at = self._camera.lookAt
        dt = self.intervention.root.dt.value
        direction = look_at - position
        direction = (
            direction
            / (direction[0] ** 2 + direction[1] ** 2 + direction[2] ** 2) ** 0.5
        )
        position += direction * velocity * dt
        self._distance -= velocity * dt
        self._camera.position = position

    def rotate(self, lao_rao_speed: float, cra_cau_speed: float):
        dt = self.intervention.root.dt.value

        look_at = self._camera.lookAt
        self._theta_x += cra_cau_speed * dt
        self._theta_z += lao_rao_speed * dt
        theta_x = self._theta_x
        theta_z = self._theta_z

        rotation_x = np.array(
            [
                [1, 0, 0],
                [0, cos(theta_x), -sin(theta_x)],
                [0, sin(theta_x), cos(theta_x)],
            ]
        )
        rotation_z = np.array(
            [
                [cos(theta_z), -sin(theta_z), 0],
                [sin(theta_z), cos(theta_z), 0],
                [0, 0, 1],
            ]
        )
        rotation = np.matmul(rotation_z, rotation_x)
        offset = np.matmul(rotation, self._initial_direction * self._distance)

        self._camera.position = look_at + np.array(offset)

        camera_rot_x = np.array([cos(theta_x / 2), sin(theta_x / 2), 0, 0])
        camera_rot_z = np.array([cos(theta_z / 2), 0, 0, sin(theta_z / 2)])

        camera_orientation = self._quat_mult(camera_rot_x, self.initial_orientation)
        camera_orientation = self._quat_mult(camera_rot_z, camera_orientation)

        self._camera.orientation = np.array(
            [
                camera_orientation[1],
                camera_orientation[2],
                camera_orientation[3],
                camera_orientation[0],
            ]
        )

    @staticmethod
    def _quat_mult(x, y):
        return np.array(
            [
                x[0] * y[0] - x[1] * y[1] - x[2] * y[2] - x[3] * y[3],
                x[0] * y[1] + x[1] * y[0] + x[2] * y[3] - x[3] * y[2],
                x[0] * y[2] - x[1] * y[3] + x[2] * y[0] + x[3] * y[1],
                x[0] * y[3] + x[1] * y[2] - x[2] * y[1] + x[3] * y[0],
            ]
        )
