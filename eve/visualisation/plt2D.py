from eve.util.binaryimage import create_walls_from_vessel_tree
from .visualisation import Visualisation

from ..vesseltree import VesselTree
from ..intervention import Intervention
from ..interimtarget import InterimTarget
from ..target import Target
import matplotlib.pyplot as plt

import matplotlib
from typing import List

import numpy as np


class PLT2D(Visualisation):
    def __init__(
        self,
        vessel_tree: VesselTree,
        intervention: Intervention,
        target: Target = None,
        interim_targets: InterimTarget = None,
        dimension_to_omit="y",
    ) -> None:
        self.vessel_tree = vessel_tree
        self.intervention = intervention
        self.target = target
        self.interim_targets = interim_targets

        self.fig, self.ax = None, None
        self.plotted_interim_targets = None
        self.target_plot = None
        self._centerline_tree = None
        self.dimension_to_omit = dimension_to_omit
        self._colors = ["black", "orange", "green", "blue"]

    def step(self) -> None:
        self._render()
        plt.draw()
        plt.pause(0.0001)
        plt.show(block=False)

    def reset(self, episode_nr: int = 0) -> None:
        if self._centerline_tree != self.vessel_tree.branches:
            self._init_vessel_tree()
            self._centerline_tree = self.vessel_tree.branches
        if self.target is not None:
            self._init_targets()
        self.step()

    def close(self):
        plt.close(self.fig)

    def _init_vessel_tree(self) -> None:
        if self.fig is None:
            self.fig, self.ax = plt.subplots()
        self.ax.clear()
        self.ax.set_xlim(
            self.vessel_tree.coordinate_space.low[0],
            self.vessel_tree.coordinate_space.high[0],
        )
        self.ax.set_ylim(
            self.vessel_tree.coordinate_space.low[2],
            self.vessel_tree.coordinate_space.high[2],
        )

        # self.ax = self.fig.gca()
        self.ax.set_aspect("equal")
        self.ax.set_axis_off()
        walls = create_walls_from_vessel_tree(
            self.vessel_tree,
            self.dimension_to_omit,
            pixel_spacing=0.05,
            contour_approx_margin=6.0,
        )
        for wall in walls:
            x = [wall.start[0], wall.end[0]]
            y = [wall.start[1], wall.end[1]]
            self.ax.plot(x, y, color="dimgrey")
        self.artists: List[matplotlib.artist.Artist] = []

    def _init_targets(self):
        if self.target_plot is not None:
            self.target_plot.remove()
        target_pos = self.target.coordinates
        if self.dimension_to_omit == "x":
            dim_del = 0
        elif self.dimension_to_omit == "z":
            dim_del = 2
        else:
            dim_del = 1
        target_pos = np.delete(target_pos, dim_del, -1)
        target_radius = self.target.threshold
        self.target_plot = plt.Circle(
            target_pos, target_radius, fill=True, color="darkorange"
        )
        self.ax.add_artist(self.target_plot)

        self.fig.canvas.draw()

    def _render(self):
        for artist in self.artists:
            artist.remove()
        self.artists = []

        trackings = self.intervention.tracking_per_device
        for i in range(len(trackings)):
            trackings[i] = [
                trackings[i],
                self._colors[i],
                self.intervention.device_diameters[i],
            ]

        trackings = sorted(trackings, key=lambda tracking: -tracking[0].shape[0])

        if self.dimension_to_omit == "x":
            dim_del = 0
        elif self.dimension_to_omit == "z":
            dim_del = 2
        else:
            dim_del = 1

        for i, tracking in enumerate(trackings):
            color = tracking[1]
            diameter = tracking[2]
            if tracking[0].shape[-1] == 3:
                tracking = np.delete(tracking[0], dim_del, axis=1)
            if i < len(trackings) - 1:
                end = tracking.shape[0] - trackings[i + 1][0].shape[0]
            else:
                end = tracking.shape[0] - 1

            for j in range(end):
                line_start = tracking[j]
                line_end = tracking[j + 1]
                x = [line_start[0], line_end[0]]
                y = [line_start[1], line_end[1]]
                line = plt.Line2D(x, y, linewidth=diameter, color=color)
                self.ax.add_line(line)
                self.artists.append(line)
