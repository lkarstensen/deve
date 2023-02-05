from typing import Tuple, Dict, Any, Optional, TypeVar, Union
import numpy as np
import gymnasium as gym

from .interimtarget import InterimTarget, InterimTargetDummy
from .pathfinder import Pathfinder, PathfinderDummy
from .intervention import Intervention
from .success import Success
from .start import Start
from .target import Target
from .visualisation import Visualisation, VisualisationDummy
from .vesseltree import VesselTree
from .observation import Observation
from .reward import Reward
from .terminal import Terminal
from .truncation import Truncation
from .info import Info, InfoDummy
from .imaging import Imaging, ImagingDummy


ObsType = TypeVar(
    "ObsType",
    np.ndarray,
    Tuple[Union[np.ndarray, Dict[str, np.ndarray]]],
    Dict[str, np.ndarray],
)
ActType = TypeVar("ActType", np.ndarray, Dict[str, np.ndarray])
RenderFrame = TypeVar("RenderFrame")


class Env(gym.Env):
    def __init__(
        self,
        vessel_tree: VesselTree,
        intervention: Intervention,
        start: Start,
        target: Target,
        success: Success,
        observation: Observation,
        reward: Reward,
        terminal: Terminal,
        truncation: Truncation,
        info: Optional[Info] = None,
        imaging: Optional[Imaging] = None,
        pathfinder: Optional[Pathfinder] = None,
        interim_target: Optional[InterimTarget] = None,
        visualisation: Optional[Visualisation] = None,
    ) -> None:

        self.vessel_tree = vessel_tree
        self.intervention = intervention
        self.success = success
        self.start = start
        self.target = target
        self.observation = observation
        self.reward = reward
        self.terminal = terminal
        self.truncation = truncation
        self.info = info or InfoDummy()
        self.imaging = imaging or ImagingDummy()
        self.pathfinder = pathfinder or PathfinderDummy()
        self.interim_target = interim_target or InterimTargetDummy()
        self.visualisation = visualisation or VisualisationDummy()

        self.episode_number = 0

    @property
    def observation_space(self) -> gym.Space:
        return self.observation.space

    @property
    def action_space(self) -> gym.Space:
        return self.intervention.action_space

    def step(
        self, action: ActType
    ) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        self.vessel_tree.step()
        self.intervention.step(action)
        self.imaging.step()
        self.pathfinder.step()
        self.target.step()
        self.interim_target.step()
        self.success.step()
        self.observation.step()
        self.reward.step()
        self.terminal.step()
        self.info.step()
        self.visualisation.step()
        return (
            self.observation(),
            self.reward.reward,
            self.terminal.terminal,
            self.truncation.truncated,
            self.info.info,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)
        self.vessel_tree.reset(self.episode_number, seed)
        self.intervention.reset(self.episode_number)
        self.start.reset(self.episode_number)
        self.target.reset(self.episode_number)
        self.pathfinder.reset(self.episode_number)
        self.interim_target.reset(self.episode_number)
        self.imaging.reset(self.episode_number)
        self.success.reset(self.episode_number)
        self.observation.reset(self.episode_number)
        self.reward.reset(self.episode_number)
        self.terminal.reset(self.episode_number)
        self.info.reset(self.episode_number)
        self.visualisation.reset(self.episode_number)
        self.episode_number += 1
        return self.observation()

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        self.visualisation.step()

    def close(self):
        self.intervention.close()
        self.visualisation.close()


class DummyEnv(Env):
    def __init__(  # pylint: disable=super-init-not-called
        self, *args, **kwds  # pylint: disable=unused-argument
    ) -> None:
        self.action_space = gym.spaces.Box(low=np.empty((1,)), high=np.empty((1,)))
        self.observation_space = gym.spaces.Box(low=np.empty((1,)), high=np.empty((1,)))

    def step(self, action: np.ndarray) -> None:
        ...

    def reset(self, *args, **kwds) -> None:  # pylint: disable=unused-argument
        ...

    def render(self) -> None:
        ...

    def close(self) -> None:
        ...
