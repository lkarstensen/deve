import logging
from typing import Dict
import Sofa

# import Sofa.Gui
import SofaRuntime  # pylint: disable=unused-import
import numpy as np


from .simulation3d import Simulation3D
from ..vesseltree import VesselTree
from .device import Device3D


class Guidewire(Simulation3D):
    def __init__(
        self,
        vessel_tree: VesselTree,
        device: Device3D,
        stop_device_at_tree_end: bool = True,
        image_frequency: float = 7.5,
        dt_simulation: float = 0.006,
        sofa_native_gui: bool = False,
    ) -> None:
        super().__init__(
            vessel_tree, image_frequency, dt_simulation, [device.velocity_limit]
        )
        self.logger = logging.getLogger(self.__module__)

        self.device = device

        self.velocity_limit = device.velocity_limit
        self.stop_device_at_tree_end = stop_device_at_tree_end

        self.sofa_native_gui = sofa_native_gui

        self.init_visual = False

        self._sofa_initialized = False
        self.sofa_initialized_2 = False
        self._loaded_mesh = None
        self.root = None
        self._beam_mechanics = None

    @property
    def tracking_ground_truth(self) -> np.ndarray:
        tracking = self._beam_mechanics.DOFs.position.value[:, 0:3][::-1]
        if np.any(np.isnan(tracking[0])):
            self.logger.warning("Tracking is NAN, resetting devices")
            self.simulation_error = True
            self._reset_sofa_devices()
            tracking = self._beam_mechanics.DOFs.position.value[:, 0:3][::-1]
        return tracking

    @property
    def tracking_per_device(self) -> Dict[Device3D, np.ndarray]:
        return {self.device: self.tracking}

    @property
    def device_lengths_inserted(self) -> Dict[Device3D, float]:
        return {self.device: self._beam_mechanics.DeployController.xtip.value[0]}

    @property
    def device_lengths_maximum(self) -> Dict[Device3D, float]:
        return {self.device: self.device.length}

    @property
    def device_rotations(self) -> Dict[Device3D, float]:
        try:
            rot = self._beam_mechanics.DeployController.rotationInstrument.value[0]
        except AttributeError:
            rot = 0.0
        return {self.device: rot}

    @property
    def device_diameters(self) -> Dict[Device3D, float]:
        return {self.device: self.device.radius * 2}

    def step(self, action: np.ndarray) -> None:
        action = np.array(action).reshape(self.action_space.shape)
        action = np.clip(action, -self.velocity_limits, self.velocity_limits)
        if (
            self.device_lengths_inserted[self.device]
            + action[0, 0] / self.image_frequency
            <= 0.0
        ):
            action[0, 0] = 0.0
        tip = self.tracking[0]
        if (
            self.stop_device_at_tree_end
            and self.vessel_tree.at_tree_end(tip)
            and action[0, 0] > 0
            and self.device_lengths_inserted[self.device] > 10
        ):
            action[0, 0] = 0.0
        self.last_action = action
        for _ in range(int((1 / self.image_frequency) / self.dt_simulation)):
            self._do_sofa_step(action)

    def reset(self, episode_nr: int = 0, seed: int = None) -> None:

        if self._loaded_mesh != self.vessel_tree.mesh_path:
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
        self._sofa_initialized = False

    def _do_sofa_step(self, action):
        if not self.sofa_initialized_2:
            Sofa.Simulation.init(self.root)
            self.sofa_initialized_2 = True
        trans = action[0, 0]
        rot = action[0, 1]
        tip = self._beam_mechanics.DeployController.xtip
        tip[0] += float(trans * self.root.dt.value)
        self._beam_mechanics.DeployController.xtip = tip
        tip_rot = self._beam_mechanics.DeployController.rotationInstrument
        tip_rot[0] += float(rot * self.root.dt.value)
        self._beam_mechanics.DeployController.rotationInstrument = tip_rot
        Sofa.Simulation.animate(self.root, self.root.dt.value)

    def _reset_sofa_devices(self):

        if not self.sofa_initialized_2:
            Sofa.Simulation.init(self.root)
            self.sofa_initialized_2 = True

        xtip = self._beam_mechanics.DeployController.xtip.value
        self._beam_mechanics.DeployController.xtip.value = xtip * 0.1
        rotInstr = self._beam_mechanics.DeployController.rotationInstrument.value
        self._beam_mechanics.DeployController.rotationInstrument.value = rotInstr * 0.0
        self._beam_mechanics.DeployController.indexFirstNode.value = 0
        Sofa.Simulation.reset(self.root)

    def _init_sofa(
        self,
        insertion_point: np.ndarray,
        insertion_direction: np.ndarray,
        mesh_path: str,
    ):

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
        self.logger.info("Sofa Initialized")
        if self.sofa_native_gui:

            Sofa.Gui.GUIManager.Init("myscene", "qglviewer")
            Sofa.Gui.GUIManager.createGUI(self.root, __file__)
            Sofa.Gui.GUIManager.SetDimension(720, 720)
            Sofa.Gui.GUIManager.MainLoop(self.root)

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
        visu_node = self.root.addChild("Visu")
        visu_node.addObject("MechanicalObject", name="Quads")
        visu_node.addObject("QuadSetTopologyContainer", name="ContainerTube")
        visu_node.addObject("QuadSetTopologyModifier", name="Modifier")
        visu_node.addObject(
            "QuadSetGeometryAlgorithms",
            name="GeomAlgo",
            template="Vec3d",
        )
        visu_node.addObject(
            "Edge2QuadTopologicalMapping",
            nbPointsOnEachCircle="10",
            radius=self.device.radius,
            flipNormals="true",
            input="@../EdgeTopology/meshLines",
            output="@ContainerTube",
        )
        visu_node.addObject(
            "AdaptiveBeamMapping",
            interpolation="@../BeamModel/BeamInterpolation",
            name="VisuMap",
            output="@Quads",
            isMechanical="false",
            input="@../BeamModel/DOFs",
            useCurvAbs="1",
            printLog="0",
        )
        visu_ogl = visu_node.addChild("VisuOgl")
        visu_ogl.addObject(
            "OglModel",
            color=self.device.color,
            quads="@../ContainerTube.quads",
            src="@../ContainerTube",
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

    def _add_vessel_tree(self, mesh_path: str):

        # if self.sofa_node is not None and hasattr(sofa_root, "vessel_tree"):
        #     sofa_root.removeChild(self.sofa_node)
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
        topo_lines = self.root.addChild("EdgeTopology")
        rest_shape_name = self.device.name + "_rest_shape"
        if not self.device.is_a_procedural_shape:
            topo_lines.addObject(
                "MeshObjLoader",
                filename=self.device.mesh_path,
                name="loader",
            )
        topo_lines.addObject(
            "WireRestShape",
            name=rest_shape_name,
            isAProceduralShape=self.device.is_a_procedural_shape,
            straightLength=self.device.straight_length,
            length=self.device.length,
            spireDiameter=self.device.spire_diameter,
            radiusExtremity=self.device.radius_extremity,
            youngModulusExtremity=self.device.young_modulus_extremity,
            massDensityExtremity=self.device.mass_density_extremity,
            radius=self.device.radius,
            youngModulus=self.device.young_modulus,
            massDensity=self.device.mass_density,
            poissonRatio=self.device.poisson_ratio,
            keyPoints=self.device.key_points,
            densityOfBeams=self.device.density_of_beams,
            numEdgesCollis=self.device.num_edges_collis,
            numEdges=self.device.num_edges,
            spireHeight=self.device.spire_height,
            printLog=True,
            template="Rigid3d",
        )
        topo_lines.addObject("EdgeSetTopologyContainer", name="meshLines")
        topo_lines.addObject("EdgeSetTopologyModifier", name="Modifier")
        topo_lines.addObject(
            "EdgeSetGeometryAlgorithms", name="GeomAlgo", template="Rigid3d"
        )
        topo_lines.addObject("MechanicalObject", name="dofTopo2", template="Rigid3d")

        beam_mechanics = self.root.addChild("BeamModel")
        beam_mechanics.addObject(
            "EulerImplicitSolver", rayleighStiffness=0.2, rayleighMass=0.1
        )
        beam_mechanics.addObject(
            "BTDLinearSolver", verification=False, subpartSolve=False, verbose=False
        )
        nx = sum(self.device.density_of_beams) + 1
        beam_mechanics.addObject(
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
        beam_mechanics.addObject(
            "MechanicalObject",
            showIndices=False,
            name="DOFs",
            template="Rigid3d",
        )
        beam_mechanics.addObject(
            "WireBeamInterpolation",
            name="BeamInterpolation",
            WireRestShape="@../EdgeTopology/" + rest_shape_name,
            radius=self.device.radius,
            printLog=False,
        )
        beam_mechanics.addObject(
            "AdaptiveBeamForceFieldAndMass",
            name="BeamForceField",
            massDensity=self.device.mass_density,
            interpolation="@BeamInterpolation",
        )

        insertion_pose = self._calculate_insertion_pose(
            insertion_point, insertion_direction
        )

        beam_mechanics.addObject(
            "InterventionalRadiologyController",
            name="DeployController",
            template="Rigid3d",
            instruments="BeamInterpolation",
            startingPos=insertion_pose,
            xtip=[0],
            printLog=True,
            rotationInstrument=[0],
            speed=0.0,
            listening=True,
            controlledInstrument=0,
        )
        beam_mechanics.addObject(
            "LinearSolverConstraintCorrection", wire_optimization="true", printLog=False
        )
        beam_mechanics.addObject("FixedConstraint", indices=0, name="FixedConstraint")
        beam_mechanics.addObject(
            "RestShapeSpringsForceField",
            points="@DeployController.indexFirstNode",
            angularStiffness=1e8,
            stiffness=1e8,
            external_points=0,
            external_rest_shape="@DOFs",
        )
        self._beam_mechanics = beam_mechanics

        beam_collis = beam_mechanics.addChild("CollisionModel")
        beam_collis.activated = True
        beam_collis.addObject("EdgeSetTopologyContainer", name="collisEdgeSet")
        beam_collis.addObject("EdgeSetTopologyModifier", name="colliseEdgeModifier")
        beam_collis.addObject("MechanicalObject", name="CollisionDOFs")
        beam_collis.addObject(
            "MultiAdaptiveBeamMapping",
            controller="../DeployController",
            useCurvAbs=True,
            printLog=False,
            name="collisMap",
        )
        beam_collis.addObject("LineCollisionModel", proximity=0.0)
        beam_collis.addObject("PointCollisionModel", proximity=0.0)
