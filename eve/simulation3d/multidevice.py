from typing import Dict, List
import logging
import Sofa

# import Sofa.Gui
import SofaRuntime  # pylint: disable=unused-import
import numpy as np

from .simulation3d import Simulation3D
from ..vesseltree import VesselTree
from .device import Device3D


class MultiDevice(Simulation3D):
    def __init__(
        self,
        vessel_tree: VesselTree,
        devices: List[Device3D],
        stop_device_at_tree_end: bool = True,
        image_frequency: float = 7.5,
        dt_simulation: float = 0.006,
        sofa_native_gui: bool = False,
    ) -> None:
        self.logger = logging.getLogger(self.__module__)

        velocity_limits = tuple(device.velocity_limit for device in devices)
        super().__init__(vessel_tree, image_frequency, dt_simulation, velocity_limits)

        self.devices = devices
        self.stop_device_at_tree_end = stop_device_at_tree_end
        self.sofa_native_gui = sofa_native_gui

        self._sofa_initialized = False
        self.sofa_initialized_2 = False
        self._loaded_mesh = None
        self.root = None
        self.instruments_combined = []
        self.device_trackings = []

        self.init_visual = False

    @property
    def tracking_ground_truth(self) -> np.ndarray:
        tracking = self.instruments_combined.DOFs.position.value[:, 0:3][::-1]
        if np.any(np.isnan(tracking[0])):
            self.logger.warning("Tracking is NAN, resetting devices")
            self.simulation_error = True
            self._reset_sofa_devices()
            tracking = self.instruments_combined.DOFs.position.value[:, 0:3][::-1]
        return tracking

    @property
    def tracking_per_device(self) -> Dict[Device3D, np.ndarray]:
        trackings = {}
        for device, device_tracking in zip(self.devices, self.device_trackings):
            tracking = device_tracking.position.value[:, 0:3][::-1]
            if not np.any(tracking):
                tracking = [
                    [
                        0.0,
                        0.0,
                        0.0,
                    ]
                ] * int(device.length + 1)
                tracking = np.array(tracking)
            trackings[device] = tracking

        return trackings

    @property
    def device_lengths_inserted(self) -> Dict[Device3D, float]:
        lengths = self.instruments_combined.m_ircontroller.xtip.value
        return dict(zip(self.devices, lengths))

    @property
    def device_lengths_maximum(self) -> Dict[Device3D, float]:
        return {device: device.length for device in self.devices}

    @property
    def device_rotations(self) -> Dict[Device3D, float]:
        try:
            rots = self.instruments_combined.m_ircontroller.rotationInstrument.value
        except AttributeError:
            rots = [0.0] * len(self.devices)
        return dict(zip(self.devices, rots))

    @property
    def device_diameters(self) -> Dict[Device3D, float]:
        return {device: device.radius * 2 for device in self.devices}

    def step(self, action: np.ndarray) -> None:
        action = np.array(action).reshape(self.action_space.shape)
        action = np.clip(action, -self.velocity_limits, self.velocity_limits)

        inserted_lengths = np.array(list(self.device_lengths_inserted.values()))
        mask = np.where(inserted_lengths + action[:, 0] / self.image_frequency <= 0.0)
        action[mask, 0] = 0.0

        tip = self.tracking[0]
        if self.vessel_tree.at_tree_end(tip) and self.stop_device_at_tree_end:

            max_length = max(inserted_lengths)
            if max_length > 10:
                dist_to_longest = -1 * inserted_lengths + max_length
                movement = action[:, 0] / self.image_frequency
                mask = movement > dist_to_longest
                action[mask, 0] = 0.0

        self.last_action = action

        for _ in range(int((1 / self.image_frequency) / self.dt_simulation)):
            self._do_sofa_step(action, inserted_lengths)

    def reset(self, episode_nr: int = 0, seed: int = None, force: bool = False) -> None:

        if self._loaded_mesh != self.vessel_tree.mesh_path or force:
            ip_pos = self.vessel_tree.insertion.position
            ip_dir = self.vessel_tree.insertion.direction
            if self._sofa_initialized:
                self._unload_sofa()
            self._init_sofa(
                insertion_point=ip_pos,
                insertion_direction=ip_dir,
                mesh_path=self.vessel_tree.mesh_path,
            )
            self._sofa_initialized = True
            self.initialized_last_reset = True
            self._loaded_mesh = self.vessel_tree.mesh_path
        else:
            self.initialized_last_reset = False
        self.simulation_error = False

    def reset_devices(self) -> None:
        if not self.initialized_last_reset:
            self._reset_sofa_devices()

    def close(self):
        self._unload_sofa()

    def _unload_sofa(self):
        Sofa.Simulation.unload(self.root)
        self.sofa_initialized_2 = False

    def _do_sofa_step(self, action: np.ndarray, inserted_lengths: np.ndarray):
        if not self.sofa_initialized_2:
            Sofa.Simulation.init(self.root)
            self.sofa_initialized_2 = True
            if self.sofa_native_gui:
                Sofa.Gui.GUIManager.Init("myscene", "qglviewer")
                Sofa.Gui.GUIManager.createGUI(self.root, __file__)
                Sofa.Gui.GUIManager.SetDimension(720, 720)
                Sofa.Gui.GUIManager.MainLoop(self.root)
        if len(self.devices) > 1:
            max_id = np.argmax(inserted_lengths)
            new_length = inserted_lengths + action[:, 0] / self.image_frequency
            new_max_id = np.argmax(new_length)
            if max_id != new_max_id:
                if abs(action[max_id, 0]) > abs(action[new_max_id, 0]):
                    action[new_max_id, 0] = 0.0
                else:
                    action[max_id, 0] = 0.0

        x_tip = self.instruments_combined.m_ircontroller.xtip
        tip_rot = self.instruments_combined.m_ircontroller.rotationInstrument
        for i in range(action.shape[0]):
            x_tip[i] += float(action[i][0] * self.root.dt.value)
            tip_rot[i] += float(action[i][1] * self.root.dt.value)
        self.instruments_combined.m_ircontroller.xtip = x_tip
        self.instruments_combined.m_ircontroller.rotationInstrument = tip_rot
        Sofa.Simulation.animate(self.root, self.root.dt.value)

    def _reset_sofa_devices(self):
        if not self.sofa_initialized_2:
            Sofa.Simulation.init(self.root)
            self.logger.info("Sofa Initialized")
            self.sofa_initialized_2 = True

        x = self.instruments_combined.m_ircontroller.xtip.value
        self.instruments_combined.m_ircontroller.xtip.value = x * 0.0
        ri = self.instruments_combined.m_ircontroller.rotationInstrument.value
        self.instruments_combined.m_ircontroller.rotationInstrument.value = ri * 0.0
        self.instruments_combined.m_ircontroller.indexFirstNode.value = 0
        Sofa.Simulation.reset(self.root)

    def _init_sofa(self, insertion_point, insertion_direction, mesh_path):
        if self.root is None:
            self.root = Sofa.Core.Node()
        self._load_plugins()
        self.root.gravity = [0.0, 0.0, 0.0]
        self.root.dt = self.dt_simulation
        self._basic_setup()
        self._add_vessel_tree(mesh_path=mesh_path)
        self._add_device(
            insertion_point=insertion_point, insertion_direction=insertion_direction
        )
        if self.init_visual:
            self._add_visual()

    def _load_plugins(self):
        self.root.addObject("RequiredPlugin", name="SofaUserInteraction")
        self.root.addObject("RequiredPlugin", name="BeamAdapter")
        self.root.addObject("RequiredPlugin", name="SofaPython3")
        self.root.addObject("RequiredPlugin", name="SofaMiscCollision")
        self.root.addObject("RequiredPlugin", name="SofaOpenglVisual")
        self.root.addObject("RequiredPlugin", name="SofaConstraint")
        self.root.addObject("RequiredPlugin", name="SofaGeneralLinearSolver")
        self.root.addObject("RequiredPlugin", name="SofaImplicitOdeSolver")
        self.root.addObject("RequiredPlugin", name="SofaLoader")
        self.root.addObject("RequiredPlugin", name="SofaBoundaryCondition")
        self.root.addObject("RequiredPlugin", name="SofaDeformable")
        self.root.addObject("RequiredPlugin", name="SofaMeshCollision")
        self.root.addObject("RequiredPlugin", name="SofaTopologyMapping")

    def _add_visual(self):
        self.vessel_object.addObject(
            "OglModel",
            src="@meshLoader",
            color=[0.5, 1.0, 1.0, 0.3],
        )

        for device in self.devices:
            visu_node = self.instrument_combined.addChild("Visu_" + device.name)
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
                radius=device.radius,
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

    def _basic_setup(self):
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
        self.root.addObject("DefaultCollisionGroupManager", name="Group")
        self.root.addObject(
            "LCPConstraintSolver",
            mu=0.1,
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
        vessel_object.addObject("MechanicalObject", src="@meshLoader")
        vessel_object.addObject(
            "MeshTopology",
            position="@meshLoader.position",
            triangles="@meshLoader.triangles",
        )

        vessel_object.addObject("TriangleCollisionModel", moving=False, simulated=False)
        vessel_object.addObject("LineCollisionModel", moving=False, simulated=False)
        self.vessel_object = vessel_object

    def _add_device(self, insertion_point, insertion_direction):

        for device in self.devices:
            topo_lines = self.root.addChild("topolines_" + device.name)
            if not device.is_a_procedural_shape:
                topo_lines.addObject(
                    "MeshObjLoader",
                    filename=device.mesh_path,
                    name="loader",
                )
            topo_lines.addObject(
                "WireRestShape",
                name="rest_shape_" + device.name,
                isAProceduralShape=device.is_a_procedural_shape,
                straightLength=device.straight_length,
                length=device.length,
                spireDiameter=device.spire_diameter,
                radiusExtremity=device.radius_extremity,
                youngModulusExtremity=device.young_modulus_extremity,
                massDensityExtremity=device.mass_density_extremity,
                radius=device.radius,
                youngModulus=device.young_modulus,
                massDensity=device.mass_density,
                poissonRatio=device.poisson_ratio,
                keyPoints=device.key_points,
                densityOfBeams=device.density_of_beams,
                numEdgesCollis=device.num_edges_collis,
                numEdges=device.num_edges,
                spireHeight=device.spire_height,
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

        instrument_combined = self.root.addChild("InstrumentCombined")
        instrument_combined.addObject(
            "EulerImplicitSolver", rayleighStiffness=0.2, rayleighMass=0.1
        )
        instrument_combined.addObject(
            "BTDLinearSolver", verification=False, subpartSolve=False, verbose=False
        )
        nx = 0
        for device in self.devices:
            nx = max([nx, sum(device.density_of_beams) + 1])

        instrument_combined.addObject(
            "RegularGridTopology",
            name="MeshLines",
            nx=nx,
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

        # rot_mat = self._rot_matrix_between_vectors(
        #     np.array([1, 0, 0]), insertion_direction
        # )
        # euler = self._rot_mat_to_euler(rot_mat)
        instrument_combined.addObject(
            "MechanicalObject",
            showIndices=False,
            name="DOFs",
            template="Rigid3d",
            # rotation=euler,
            # translation=insertion_point,
        )
        x_tip = []
        rotations = []
        interpolations = ""

        for device in self.devices:
            wire_rest_shape = (
                "@../topolines_" + device.name + "/rest_shape_" + device.name
            )
            instrument_combined.addObject(
                "WireBeamInterpolation",
                name="Interpol_" + device.name,
                WireRestShape=wire_rest_shape,
                radius=device.radius,
                printLog=False,
            )
            instrument_combined.addObject(
                "AdaptiveBeamForceFieldAndMass",
                name="ForceField_" + device.name,
                massDensity=device.mass_density,
                interpolation="@Interpol_" + device.name,
            )
            x_tip.append(0.0)
            rotations.append(0.0)
            interpolations += "Interpol_" + device.name + " "
        x_tip[0] += 0.1
        interpolations = interpolations[:-1]

        insertion_pose = self._calculate_insertion_pose(
            insertion_point, insertion_direction
        )

        instrument_combined.addObject(
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
        instrument_combined.addObject(
            "LinearSolverConstraintCorrection", wire_optimization="true", printLog=False
        )
        instrument_combined.addObject(
            "FixedConstraint", indices=0, name="FixedConstraint"
        )
        instrument_combined.addObject(
            "RestShapeSpringsForceField",
            points="@m_ircontroller.indexFirstNode",
            angularStiffness=1e8,
            stiffness=1e8,
            external_points=0,
            external_rest_shape="@DOFs",
        )
        self.instruments_combined = instrument_combined

        beam_collis = instrument_combined.addChild("CollisionModel")
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
        self.instrument_combined = instrument_combined
        # self.device_trackings = []
        # for device in self.devices:
        #     mesh_lines = "@../../topolines_" + device.name + "/meshLines_" + device.name

        #     tracking_node = instrument_combined.addChild("tracking_" + device.name)
        #     tracking_node.activated = True
        #     if not device.is_a_procedural_shape:
        #         tracking_node.addObject(
        #             "MeshObjLoader",
        #             filename=device.mesh_path,
        #             name="loader",
        #         )

        #     tracking_node.addObject(
        #         "WireRestShape",
        #         name="rest_shape_" + device.name,
        #         isAProceduralShape=device.is_a_procedural_shape,
        #         straightLength=device.straight_length,
        #         length=device.length,
        #         spireDiameter=device.spire_diameter,
        #         radiusExtremity=device.radius_extremity,
        #         youngModulusExtremity=device.young_modulus_extremity,
        #         massDensityExtremity=device.mass_density_extremity,
        #         radius=device.radius,
        #         youngModulus=device.young_modulus,
        #         massDensity=device.mass_density,
        #         poissonRatio=device.poisson_ratio,
        #         keyPoints=device.key_points,
        #         densityOfBeams=device.density_of_beams,
        #         numEdgesCollis=device.num_edges_collis,
        #         numEdges=int(device.length) + 1,
        #         spireHeight=device.spire_height,
        #         printLog=True,
        #         template="Rigid3d",
        #     )

        #     tracking_node.addObject("EdgeSetTopologyContainer", name="tracking_lines")
        #     tracking_node.addObject("EdgeSetTopologyModifier", name="Modifier")
        #     tracking_node.addObject(
        #         "EdgeSetGeometryAlgorithms", name="GeomAlgo", template="Rigid3d"
        #     )
        #     tracking_dof = tracking_node.addObject(
        #         "MechanicalObject", name="TrackingDOFs", template="Rigid3d"
        #     )
        #     tracking_node.addObject(
        #         "AdaptiveBeamMapping",
        #         interpolation="@../Interpol_" + device.name,
        #         name="TrackingMap",
        #         output="@TrackingDOFs",
        #         isMechanical="false",
        #         input="@../DOFs",
        #         useCurvAbs="1",
        #         printLog="0",
        #     )
        #     self.device_trackings.append(tracking_dof)

    def _rot_matrix_between_vectors(self, vector0, vector1):
        vector1 = vector1 / np.linalg.norm(vector1)
        vector0 = vector0 / np.linalg.norm(vector0)

        v = np.cross(vector0, vector1)
        s = np.linalg.norm(v)
        vx = np.array(
            [
                [0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0],
            ]
        )
        c = np.dot(vector0, vector1)
        I = np.eye(3)
        R = I + vx + np.matmul(vx, vx) * ((1 - c) / s**2)
        return R

    def _rot_mat_to_euler(self, rot_mat):

        sy = np.sqrt(rot_mat[0, 0] * rot_mat[0, 0] + rot_mat[1, 0] * rot_mat[1, 0])

        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(rot_mat[2, 1], rot_mat[2, 2])
            y = np.arctan2(-rot_mat[2, 0], sy)
            z = np.arctan2(rot_mat[1, 0], rot_mat[0, 0])
        else:
            x = np.arctan2(-rot_mat[1, 2], rot_mat[1, 1])
            y = np.arctan2(-rot_mat[2, 0], sy)
            z = 0

        x = np.rad2deg(x)
        y = np.rad2deg(y)
        z = np.rad2deg(z)
        return np.array([x, y, z])
