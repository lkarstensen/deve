from copy import deepcopy
import importlib
import math
import os
from typing import List, Optional, Tuple
import logging
import numpy as np
from .simulation import Simulation
from ..device import Device


class SofaBeamAdapter(Simulation):
    def __init__(
        self,
        friction: float = 0.1,
        dt_simulation: float = 0.006,
    ) -> None:
        self.logger = logging.getLogger(self.__module__)

        self.friction = friction
        self.dt_simulation = dt_simulation

        self.root = None
        self.camera = None
        self.target_node = None
        self.interim_target_node = None
        self.interim_targets = []
        self.simulation_error = False

        self.init_visual_nodes = False
        self.display_size = (1, 1)
        self.target_size = 1
        self.interim_target_size = 1

        self._vessel_object = None
        self._instruments_combined = None

        self._sofa = None
        self._sofa_runtime = None

        self._insertion_point = np.empty(())
        self._insertion_direction = np.empty(())
        self._mesh_path: str = None
        self._reset_add_visual: bool = None
        self._display_size = None
        self._coords_high = np.empty(())
        self._coords_low = np.empty(())
        self._target_size = None
        self._vessel_visual_path: str = None
        self._rng = np.random.default_rng()
        self._dof_positions = None
        self._inserted_lengths = None
        self._rotations = None

    @property
    def dof_positions(self) -> np.ndarray:
        return self._dof_positions

    @property
    def inserted_lengths(self) -> List[float]:
        return self._inserted_lengths

    @property
    def rotations(self) -> List[float]:
        return self._rotations

    def close(self):
        self._unload_simulation()

    def _unload_simulation(self):
        if self.root is not None:
            self._sofa.Simulation.unload(self.root)

    def step(self, action: np.ndarray, duration: float):
        n_steps = int(duration / self.dt_simulation)
        for _ in range(n_steps):
            inserted_lengths = self.inserted_lengths

            if len(inserted_lengths) > 1:
                max_id = np.argmax(inserted_lengths)
                new_length = inserted_lengths + action[:, 0] * self.dt_simulation
                new_max_id = np.argmax(new_length)
                if max_id != new_max_id:
                    if abs(action[max_id, 0]) > abs(action[new_max_id, 0]):
                        action[new_max_id, 0] = 0.0
                    else:
                        action[max_id, 0] = 0.0

            x_tip = self._instruments_combined.m_ircontroller.xtip
            tip_rot = self._instruments_combined.m_ircontroller.rotationInstrument
            for i in range(action.shape[0]):
                x_tip[i] += float(action[i][0] * self.root.dt.value)
                tip_rot[i] += float(action[i][1] * self.root.dt.value)
            self._instruments_combined.m_ircontroller.xtip = x_tip
            self._instruments_combined.m_ircontroller.rotationInstrument = tip_rot
            self._sofa.Simulation.animate(self.root, self.root.dt.value)
        self._update_properties()

    def reset_devices(self):
        x = self._instruments_combined.m_ircontroller.xtip.value
        self._instruments_combined.m_ircontroller.xtip.value = x * 0.0
        ri = self._instruments_combined.m_ircontroller.rotationInstrument.value
        ri = self._rng.random(ri.shape) * 2 * np.pi
        self._instruments_combined.m_ircontroller.rotationInstrument.value = ri
        self._instruments_combined.m_ircontroller.indexFirstNode.value = 0
        self._sofa.Simulation.reset(self.root)
        self._update_properties()

    def reset(
        self,
        insertion_point,
        insertion_direction,
        mesh_path,
        devices: List[Device],
        coords_high: Optional[Tuple[float, float, float]] = None,
        coords_low: Optional[Tuple[float, float, float]] = None,
        vessel_visual_path: Optional[str] = None,
        seed: int = None,
    ):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if self._sofa is None:
            self._sofa = importlib.import_module("Sofa")
        if self._sofa_runtime is None:
            self._sofa_runtime = importlib.import_module("SofaRuntime")

        if (
            self.root is None
            or np.any(insertion_point != self._insertion_point)
            or np.any(insertion_direction != self._insertion_direction)
            or mesh_path != self._mesh_path
            or vessel_visual_path != self._vessel_visual_path
            or np.any(coords_high != self._coords_high)
            or np.any(coords_low != self._coords_low)
            # or self.init_visual_nodes
        ):
            if self.root is None:
                self.root = self._sofa.Core.Node()
            else:
                self._unload_simulation()

            self.root.gravity = [0.0, 0.0, 0.0]
            self.root.dt = self.dt_simulation
            self._load_plugins()
            self._basic_setup(self.friction)
            self._add_vessel_tree(mesh_path=mesh_path)
            self._add_devices(
                devices=devices,
                insertion_point=insertion_point,
                insertion_direction=insertion_direction,
            )
            if self.init_visual_nodes:
                self._add_visual(
                    self.display_size,
                    coords_low,
                    coords_high,
                    self.target_size,
                    self.interim_target_size,
                    devices=devices,
                    vessel_visual_path=vessel_visual_path,
                )

            self._sofa.Simulation.init(self.root)
            self._insertion_point = insertion_point
            self._insertion_direction = insertion_direction
            self._mesh_path = mesh_path
            self._coords_high = coords_high
            self._coords_low = coords_low
            self._vessel_visual_path = vessel_visual_path
            self.simulation_error = False
            self.logger.debug("Sofa Initialized")
        self._update_properties()

    def _update_properties(self) -> None:
        tracking = self._instruments_combined.DOFs.position.value[:, 0:3][::-1]
        if np.any(np.isnan(tracking[0])):
            self.logger.warning("Tracking is NAN, resetting devices")
            self.simulation_error = True
            self.reset_devices()
            tracking = self._instruments_combined.DOFs.position.value[:, 0:3][::-1]
        self._dof_positions = deepcopy(tracking)
        self._inserted_lengths = deepcopy(
            self._instruments_combined.m_ircontroller.xtip.value
        )
        self._rotations = deepcopy(
            self._instruments_combined.m_ircontroller.rotationInstrument.value
        )

    def _load_plugins(self):
        self.root.addObject(
            "RequiredPlugin",
            pluginName="\
            BeamAdapter\
            Sofa.Component.AnimationLoop\
            Sofa.Component.Collision.Detection.Algorithm\
            Sofa.Component.Collision.Detection.Intersection\
            Sofa.Component.LinearSolver.Direct\
            Sofa.Component.IO.Mesh\
            Sofa.Component.ODESolver.Backward\
            Sofa.Component.Constraint.Lagrangian.Correction\
            Sofa.Component.Topology.Mapping",
        )

    def _basic_setup(self, friction: float):
        self.root.addObject("FreeMotionAnimationLoop")
        self.root.addObject("DefaultPipeline", draw="0", depth="6", verbose="1")
        self.root.addObject("BruteForceBroadPhase")
        self.root.addObject("BVHNarrowPhase")
        self.root.addObject(
            "LocalMinDistance",
            contactDistance=0.3,
            alarmDistance=0.5,
            angleCone=0.02,
            name="localmindistance",
        )
        self.root.addObject(
            "DefaultContactManager", response="FrictionContactConstraint"
        )
        self.root.addObject(
            "LCPConstraintSolver",
            mu=friction,
            tolerance=1e-4,
            maxIt=2000,
            name="LCP",
            build_lcp=False,
        )

    def _add_vessel_tree(self, mesh_path):
        vessel_object = self.root.addChild("vesselTree")
        vessel_object.addObject(
            "MeshObjLoader",
            filename=mesh_path,
            flipNormals=False,
            name="meshLoader",
        )
        vessel_object.addObject(
            "MeshTopology",
            position="@meshLoader.position",
            triangles="@meshLoader.triangles",
        )
        vessel_object.addObject("MechanicalObject", name="dofs", src="@meshLoader")
        vessel_object.addObject("TriangleCollisionModel", moving=False, simulated=False)
        vessel_object.addObject("LineCollisionModel", moving=False, simulated=False)
        self._vessel_object = vessel_object

    def _add_devices(self, devices: List[Device], insertion_point, insertion_direction):
        for device in devices:
            sofa_device = device.sofa_device
            topo_lines = self.root.addChild("topolines_" + device.name)
            if not sofa_device.is_a_procedural_shape:
                topo_lines.addObject(
                    "MeshObjLoader",
                    filename=device.sofa_device.mesh_path,
                    name="loader",
                )
            topo_lines.addObject(
                "WireRestShape",
                name="rest_shape_" + device.name,
                isAProceduralShape=sofa_device.is_a_procedural_shape,
                straightLength=sofa_device.straight_length,
                length=sofa_device.length,
                spireDiameter=sofa_device.spire_diameter,
                radiusExtremity=sofa_device.radius_extremity,
                youngModulusExtremity=sofa_device.young_modulus_extremity,
                massDensityExtremity=sofa_device.mass_density_extremity,
                radius=sofa_device.radius,
                youngModulus=sofa_device.young_modulus,
                massDensity=sofa_device.mass_density,
                poissonRatio=sofa_device.poisson_ratio,
                keyPoints=sofa_device.key_points,
                densityOfBeams=sofa_device.density_of_beams,
                numEdgesCollis=sofa_device.num_edges_collis,
                numEdges=sofa_device.num_edges,
                spireHeight=sofa_device.spire_height,
                printLog=True,
                template="Rigid3d",
            )
            topo_lines.addObject(
                "EdgeSetTopologyContainer", name="meshLines_" + device.name
            )
            topo_lines.addObject("EdgeSetTopologyModifier", name="Modifier")
            topo_lines.addObject(
                "EdgeSetGeometryAlgorithms", name="GeomAlgo", template="Rigid3d"
            )
            topo_lines.addObject(
                "MechanicalObject", name="dofTopo_" + device.name, template="Rigid3d"
            )

        instruments_combined = self.root.addChild("InstrumentCombined")
        instruments_combined.addObject(
            "EulerImplicitSolver", rayleighStiffness=0.2, rayleighMass=0.1
        )
        instruments_combined.addObject(
            "BTDLinearSolver", verification=False, subpartSolve=False, verbose=False
        )
        nx = 0
        for device in devices:
            nx = sum([nx, sum(device.sofa_device.density_of_beams)])

        instruments_combined.addObject(
            "RegularGridTopology",
            name="MeshLines",
            nx=nx + 1,
            ny=1,
            nz=1,
            xmax=1.0,
            xmin=0.0,
            ymin=0,
            ymax=0,
            zmax=1,
            zmin=1,
            p0=[0, 0, 0],
        )
        instruments_combined.addObject(
            "MechanicalObject",
            showIndices=False,
            name="DOFs",
            template="Rigid3d",
        )
        x_tip = []
        rotations = []
        interpolations = ""

        for device in devices:
            wire_rest_shape = (
                "@../topolines_" + device.name + "/rest_shape_" + device.name
            )
            instruments_combined.addObject(
                "WireBeamInterpolation",
                name="Interpol_" + device.name,
                WireRestShape=wire_rest_shape,
                radius=device.sofa_device.radius,
                printLog=False,
            )
            instruments_combined.addObject(
                "AdaptiveBeamForceFieldAndMass",
                name="ForceField_" + device.name,
                massDensity=device.sofa_device.mass_density,
                interpolation="@Interpol_" + device.name,
            )
            x_tip.append(0.0)
            rotations.append(self._rng.random() * math.pi * 2)
            interpolations += "Interpol_" + device.name + " "
        x_tip[0] += 0.1
        interpolations = interpolations[:-1]

        insertion_pose = self._calculate_insertion_pose(
            insertion_point, insertion_direction
        )

        instruments_combined.addObject(
            "InterventionalRadiologyController",
            name="m_ircontroller",
            template="Rigid3d",
            instruments=interpolations,
            startingPos=insertion_pose,
            xtip=x_tip,
            printLog=True,
            rotationInstrument=rotations,
            speed=0.0,
            listening=True,
            controlledInstrument=0,
        )

        instruments_combined.addObject(
            "LinearSolverConstraintCorrection", wire_optimization="true", printLog=False
        )
        instruments_combined.addObject(
            "FixedConstraint", indices=0, name="FixedConstraint"
        )
        instruments_combined.addObject(
            "RestShapeSpringsForceField",
            points="@m_ircontroller.indexFirstNode",
            angularStiffness=1e8,
            stiffness=1e8,
            external_points=0,
            external_rest_shape="@DOFs",
        )
        self._instruments_combined = instruments_combined

        beam_collis = instruments_combined.addChild("CollisionModel")
        beam_collis.activated = True
        beam_collis.addObject("EdgeSetTopologyContainer", name="collisEdgeSet")
        beam_collis.addObject("EdgeSetTopologyModifier", name="colliseEdgeModifier")
        beam_collis.addObject("MechanicalObject", name="CollisionDOFs")
        beam_collis.addObject(
            "MultiAdaptiveBeamMapping",
            controller="../m_ircontroller",
            useCurvAbs=True,
            printLog=False,
            name="collisMap",
        )
        beam_collis.addObject("LineCollisionModel", proximity=0.0)
        beam_collis.addObject("PointCollisionModel", proximity=0.0)

    def _add_visual(
        self,
        display_size: Tuple[int, int],
        coords_low: Tuple[float, float, float],
        coords_high: Tuple[float, float, float],
        target_size: float,
        interim_target_size: float,
        devices: List[Device],
        vessel_visual_path: Optional[str] = None,
    ):
        coords_low = np.array(coords_low)
        coords_high = np.array(coords_high)
        self.root.addObject(
            "RequiredPlugin",
            pluginName="\
            Sofa.GL.Component.Rendering3D\
            Sofa.GL.Component.Shader",
        )

        # Vessel Tree
        if vessel_visual_path is None:
            self._vessel_object.addObject(
                "OglModel",
                src="@meshLoader",
                color=[1.0, 0.0, 0.0, 0.3],
            )
        else:
            visu_vessel = self._vessel_object.addChild("Visual Vessel")
            visu_vessel.addObject(
                "MeshObjLoader", name="loader", filename=vessel_visual_path
            )
            visu_vessel.addObject("MechanicalObject", name="visu")
            visu_vessel.addObject(
                "OglModel", name="Visu", src="@loader", color=[1.0, 0.0, 0.0, 0.3]
            )
            visu_vessel.addObject(
                "BarycentricMapping", input="@../dofs", output="@Visu"
            )

        # Devices
        for device in devices:
            visu_node = self._instruments_combined.addChild("Visu_" + device.name)
            visu_node.activated = True
            visu_node.addObject("MechanicalObject", name="Quads")
            visu_node.addObject(
                "QuadSetTopologyContainer", name="Container_" + device.name
            )
            visu_node.addObject("QuadSetTopologyModifier", name="Modifier")
            visu_node.addObject(
                "QuadSetGeometryAlgorithms",
                name="GeomAlgo",
                template="Vec3d",
            )
            mesh_lines = "@../../topolines_" + device.name + "/meshLines_" + device.name
            visu_node.addObject(
                "Edge2QuadTopologicalMapping",
                nbPointsOnEachCircle=10,
                radius=device.sofa_device.radius,
                flipNormals="true",
                input=mesh_lines,
                output="@Container_" + device.name,
            )
            visu_node.addObject(
                "AdaptiveBeamMapping",
                interpolation="@../Interpol_" + device.name,
                name="VisuMap_" + device.name,
                output="@Quads",
                isMechanical="false",
                input="@../DOFs",
                useCurvAbs="1",
                printLog="0",
            )
            visu_ogl = visu_node.addChild("VisuOgl")
            visu_ogl.activated = True
            visu_ogl.addObject(
                "OglModel",
                color=device.color,
                quads="@../Container_" + device.name + ".quads",
                material="texture Ambient 1 0.2 0.2 0.2 0.0 Diffuse 1 1.0 1.0 1.0 1.0 Specular 1 1.0 1.0 1.0 1.0 Emissive 0 0.15 0.05 0.05 0.0 Shininess 1 20",
                name="Visual",
            )
            visu_ogl.addObject(
                "IdentityMapping",
                input="@../Quads",
                output="@Visual",
            )

        # Target
        # TODO: Fix necessary translation of ogl_model. Maybe unite_sphere.obj with center in origin?
        file_dir = os.path.dirname(os.path.realpath(__file__))
        mesh_path = os.path.join(file_dir, "util", "unit_sphere.stl")
        target_node = self.root.addChild("main_target")
        target_node.addObject(
            "MeshSTLLoader",
            name="loader",
            triangulate=True,
            filename=mesh_path,
            scale=target_size,
            translation=[0, 0, 0],
        )
        target_node.addObject(
            "MechanicalObject",
            src="@loader",
            translation=(0, 0, 0),
            template="Rigid3d",
            name="MechanicalObject",
        )
        size_half = target_size / 2
        target_node.addObject(
            "OglModel",
            src="@loader",
            color=[0.0, 0.9, 0.5, 0.8],
            translation=[0, 0, -size_half],
            material="texture Ambient 1 0.2 0.2 0.2 0.0 Diffuse 1 1.0 1.0 1.0 1.0 Specular 1 1.0 1.0 1.0 1.0 Emissive 0 0.15 0.05 0.05 0.0 Shininess 1 20",
            name="ogl_model",
        )
        target_node.addObject("RigidMapping", input="@MechanicalObject")
        self.target_node = target_node

        self.interim_targets = []
        interim_target_node = self.root.addChild("interim_target")
        interim_target_node.addObject(
            "MeshSTLLoader",
            name="loader",
            triangulate=True,
            filename=mesh_path,
            scale=interim_target_size,
            translation=[0, 0, 0],
        )
        interim_target_node.addObject(
            "MechanicalObject",
            src="@loader",
            translation=(9999, 0, 0),
            template="Rigid3d",
            name="MechanicalObject",
        )
        self.interim_target_node = interim_target_node
        for i in range(100):
            interim_node = self.interim_target_node.addChild(f"interim_node_{i}")

            interim_node.addObject(
                "OglModel",
                src="@../loader",
                color=[0.0, 0.9, 0.5, 0.2],
                translation=[0.0, 0.0, 0.0],
                material="texture Ambient 1 0.2 0.2 0.2 0.0 Diffuse 1 1.0 1.0 1.0 1.0 Specular 1 1.0 1.0 1.0 1.0 Emissive 0 0.15 0.05 0.05 0.0 Shininess 1 20",
                name="ogl_model",
            )
            self.interim_targets.append(interim_node)

        # Camera
        # TODO: Find out how to manipulate background. BackgroundSetting doesn't seem to work
        # self.root.addObject("BackgroundSetting", color=(0.5, 0.5, 0.5, 1.0))
        self.root.addObject("DefaultVisualManagerLoop")
        self.root.addObject(
            "VisualStyle",
            displayFlags="showVisualModels\
                hideBehaviorModels\
                hideCollisionModels\
                hideWireframe\
                hideMappings\
                hideForceFields",
        )
        self.root.addObject("LightManager")
        self.root.addObject("DirectionalLight", direction=[0, -1, 0])
        self.root.addObject("DirectionalLight", direction=[0, 1, 0])

        look_at = (coords_high + coords_low) * 0.5
        distance_coefficient = 1.5
        distance = np.linalg.norm(look_at - coords_low) * distance_coefficient
        position = look_at + np.array([0.0, -distance, 0.0])
        scene_radius = np.linalg.norm(coords_high - coords_low)
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

        self.camera = self.root.addObject(
            "Camera",
            name="camera",
            lookAt=look_at,
            position=position,
            fieldOfView=field_of_view,
            widthViewport=display_size[0],
            heightViewport=display_size[1],
            zNear=z_near,
            zFar=z_far,
            fixedLookAt=False,
        )

    def add_interim_targets(self, positions: List[Tuple[float, float, float]]):
        n_targets = min(len(positions), len(self.interim_targets))
        for i in range(n_targets):
            position = tuple(positions[i])
            self.interim_targets[i].ogl_model.translation = position
        targets_to_remove = self.interim_targets[i + 1 :]
        for target in targets_to_remove:
            self.remove_interim_target(target)

        self._sofa.Simulation.init(self.interim_target_node)
        return self.interim_targets.copy()

    def remove_interim_target(self, interim_target):
        self.interim_target_node.removeChild(interim_target)
        self.interim_targets.remove(interim_target)

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
