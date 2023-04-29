from typing import Optional, Tuple
from enum import Enum
import numpy as np

from .vesseltree import VesselTree, Insertion, Branch, gym
from .util.branch import calc_branching, scale, rotate, fill_axis_with_dummy_value
from .aorticarcharteries_old import (
    aorta_generator,
    brachiocephalic_trunk_static,
    left_common_carotid,
    left_common_carotid_II,
    left_subclavian,
    left_subclavian_IV,
    left_subclavian_VI,
    right_common_carotid,
    right_common_carotid_V,
    right_common_carotid_VII,
    right_subclavian,
    right_subclavian_IV,
    right_subclavian_V,
    right_subclavian_VI,
    common_origin_VI,
)
from .util import calc_insertion_from_branch_start
from .util.meshing import generate_temp_mesh

COORD_SPACE_BUFFER = 0.00


class ArchType(str, Enum):
    I = "I"
    II = "II"
    IV = "IV"
    Va = "Va"
    Vb = "Vb"
    VI = "VI"
    VII = "VII"


class AorticArchOld(VesselTree):
    def __init__(
        self,
        arch_type: ArchType = ArchType.I,
        seed: Optional[int] = None,
        rotate_yzx_deg: Optional[Tuple[float, float, float]] = None,
        scale_xyzd: Optional[Tuple[float, float, float, float]] = None,
        omit_axis: Optional[str] = None,
    ) -> None:
        self.arch_type = arch_type
        self.seed = seed or np.random.randint(0, 10**10)
        self.rotate_yzx_deg = rotate_yzx_deg
        self.scale_xyzd = scale_xyzd
        self.omit_axis = omit_axis

        self._mesh_path = None
        self.centerline_coordinates = None
        self.branches = None
        self.branching_points = None
        self.insertion = None

        branches = self._generate_branches()
        self.coordinate_space = self._calc_coord_space(branches)
        self.coordinate_space_episode = self.coordinate_space

    @property
    def mesh_path(self) -> str:
        if self._mesh_path is None:
            self._mesh_path = generate_temp_mesh(self.branches, "aorticarch", 0.99)
        return self._mesh_path

    def reset(self, episode_nr=0, seed: int = None) -> None:
        if self.branches is None:
            branches = self._generate_branches()
            insertion_point, ip_dir = calc_insertion_from_branch_start(branches[0])

            self.coordinate_space = self._calc_coord_space(branches)
            centerline_coordinates = [branch.coordinates for branch in branches]
            self.centerline_coordinates = np.concatenate(centerline_coordinates)
            self.branches = branches
            self.insertion = Insertion(insertion_point, ip_dir)
            self.branching_points = calc_branching(branches)
            self._mesh_path = None

    def _calc_coord_space(self, branches):
        branch_highs = [branch.high for branch in branches]
        high = np.max(branch_highs, axis=0)
        branch_lows = [branch.low for branch in branches]
        low = np.min(branch_lows, axis=0)
        coord_space = gym.spaces.Box(low=low, high=high)
        return coord_space

    def _generate_branches(self):
        branches = self._generate_unmodified_branches()
        if self.rotate_yzx_deg is not None:
            branches = rotate(branches, self.rotate_yzx_deg)
        if self.scale_xyzd is not None:
            branches = scale(branches, self.scale_xyzd)
        if self.omit_axis is not None:
            branches = fill_axis_with_dummy_value(branches, self.omit_axis)
        return branches

    def _generate_unmodified_branches(self) -> Tuple[Branch]:
        rng = np.random.default_rng(self.seed)
        normal = rng.normal

        aorta_resolution = 2
        bct_resolution = 1
        aorta, _ = aorta_generator(aorta_resolution, rng)
        if self.arch_type == ArchType.I:
            branches = self._create_type_I(rng, normal, aorta_resolution, aorta)

        elif self.arch_type == ArchType.II:
            branches = self._create_type_II(
                rng, normal, aorta_resolution, bct_resolution, aorta
            )

        elif self.arch_type == ArchType.IV:
            branches = self._create_type_IV(
                rng, normal, aorta_resolution, bct_resolution, aorta
            )

        elif self.arch_type == ArchType.Va:
            branches = self._create_Va(rng, normal, aorta_resolution, aorta)

        elif self.arch_type == ArchType.Vb:
            branches = self._create_Vb(rng, normal, aorta_resolution, aorta)

        # TODO: Change this after IJCARS23
        # elif self.arch_type == ArchType.V:
        #     if rng.random() < 0.5:
        #         branches = self._create_Va(rng, normal, aorta_resolution, aorta)
        #     else:
        #         branches = self._create_Vb(rng, normal, aorta_resolution, aorta)
        elif self.arch_type == ArchType.VI:
            branches = self._create_VI(rng, normal, aorta_resolution, aorta)

        elif self.arch_type == ArchType.VII:
            branches = self._create_VII(rng, normal, aorta_resolution, aorta)
        else:
            raise ValueError(f"{self.arch_type=} not supportet.")

        return tuple(branches)

    def _create_VII(self, rng, normal, aorta_resolution, aorta):
        distance_aorta_end_rsca = normal(38, 3.5)
        idx = int(np.round(distance_aorta_end_rsca / aorta_resolution, 0))
        rsa, _ = right_subclavian_IV(aorta.coordinates[-idx], 1, rng)

        distance_rsca_rcca = normal(20, 3)
        idx += int(np.round(distance_rsca_rcca / aorta_resolution, 0))
        rcca, _ = right_common_carotid_VII(aorta.coordinates[-idx], 1, rng)

        distance_rcca_lcca = normal(20, 3)
        idx += int(np.round(distance_rcca_lcca / aorta_resolution, 0))
        lcca, _ = left_common_carotid(aorta.coordinates[-idx], 1, rng)

        distance_lcca_lsca = normal(27, 3)
        idx += int(np.round(distance_lcca_lsca / aorta_resolution, 0))
        lsa, _ = left_subclavian(aorta.coordinates[-idx], 1, rng)
        return [aorta, rcca, rsa, lcca, lsa]

    def _create_VI(self, rng, normal, aorta_resolution, aorta):
        distance_aorta_end_bct = normal(36, 3)
        idx = int(np.round(distance_aorta_end_bct / aorta_resolution, 0))
        bct, bct_chs_points = brachiocephalic_trunk_static(
            aorta.coordinates[-idx], 1, rng
        )

        rcca, _ = right_common_carotid_V(
            bct.coordinates[-1], bct_chs_points[-1], 1, rng
        )
        lcca, _ = left_common_carotid_II(bct.coordinates[-1], 1, rng)

        distance_bct_co = normal(50, 1.5)
        idx += int(np.round(distance_bct_co / aorta_resolution, 0))
        co, co_chs_points = common_origin_VI(aorta.coordinates[-idx], 1, rng)

        rsa, _ = right_subclavian_VI(co.coordinates[-1], co_chs_points[-1], 1, rng)
        lsa, _ = left_subclavian_VI(co.coordinates[-1], 1, rng)
        return [aorta, bct, co, rcca, rsa, lcca, lsa]

    def _create_Vb(self, rng, normal, aorta_resolution, aorta):
        distance_aorta_end_rcca = normal(50, 2.5)
        idx = int(np.round(distance_aorta_end_rcca / aorta_resolution, 0))
        rcca, _ = right_common_carotid_VII(aorta.coordinates[-idx], 1, rng)

        distance_rcca_lcca = normal(22, 3)
        idx += int(np.round(distance_rcca_lcca / aorta_resolution, 0))
        lcca, _ = left_common_carotid(aorta.coordinates[-idx], 1, rng)

        distance_lcca_lsca = normal(26, 2.5)
        idx += int(np.round(distance_lcca_lsca / aorta_resolution, 0))
        lsa, _ = left_subclavian(aorta.coordinates[-idx], 1, rng)

        distance_lsca_rsca = normal(20, 1)
        idx += int(np.round(distance_lsca_rsca / aorta_resolution, 0))
        rsa, _ = right_subclavian_V(aorta.coordinates[-idx], 1, rng)
        return [aorta, rcca, rsa, lcca, lsa]

    def _create_Va(self, rng, normal, aorta_resolution, aorta):
        distance_aorta_end_bct = normal(36, 3)
        idx = int(np.round(distance_aorta_end_bct / aorta_resolution, 0))
        bct, bct_chs_points = brachiocephalic_trunk_static(
            aorta.coordinates[-idx], 1, rng
        )

        rcca, _ = right_common_carotid_V(
            bct.coordinates[-1], bct_chs_points[-1], 1, rng
        )
        lcca, _ = left_common_carotid_II(bct.coordinates[-1], 1, rng)

        distance_bct_lsca = normal(41, 2.5)
        idx += int(np.round(distance_bct_lsca / aorta_resolution, 0))
        lsa, _ = left_subclavian(aorta.coordinates[-idx], 1, rng)

        distance_lsca_rsca = normal(20, 1)
        idx += int(np.round(distance_lsca_rsca / aorta_resolution, 0))
        rsa, _ = right_subclavian_V(aorta.coordinates[-idx], 1, rng)
        return [aorta, bct, rcca, rsa, lcca, lsa]

    def _create_type_IV(self, rng, normal, aorta_resolution, bct_resolution, aorta):
        distance_aorta_end_rsca = normal(42, 5)
        idx = int(np.round(distance_aorta_end_rsca / aorta_resolution, 0))
        rsa, _ = right_subclavian_IV(aorta.coordinates[-idx], 1, rng)

        distance_rsca_co = normal(20, 4)
        idx += int(np.round(distance_rsca_co / aorta_resolution, 0))
        co, _ = common_origin_VI(aorta.coordinates[-idx], bct_resolution, rng)

        rcca, _ = right_common_carotid(co.coordinates[-1], 1, rng)
        lcca, _ = left_common_carotid_II(co.coordinates[-1], 1, rng)

        distance_bct_lsca = normal(38, 3)
        idx += int(np.round(distance_bct_lsca / aorta_resolution, 0))
        lsa, _ = left_subclavian_IV(aorta.coordinates[-idx], 1, rng)
        return [aorta, co, rcca, rsa, lcca, lsa]

    def _create_type_II(self, rng, normal, aorta_resolution, bct_resolution, aorta):
        distance_aorta_end_bct = normal(36, 3)
        idx = int(np.round(distance_aorta_end_bct / aorta_resolution, 0))
        bct, bct_chs_points = brachiocephalic_trunk_static(
            aorta.coordinates[-idx], bct_resolution, rng
        )
        rsa, _ = right_subclavian(bct.coordinates[-1], bct_chs_points[-1], 1, rng)
        rcca, _ = right_common_carotid(bct.coordinates[-1], 1, rng)

        distance_aorta_lcca = normal(bct.length * (2 / 3), bct.length * (3 / 10) / 3)
        distance_aorta_lcca = abs(distance_aorta_lcca)  # if distance < 0
        distance_aorta_lcca = min(
            bct.length, distance_aorta_lcca
        )  # if distance longer than bct
        lcca_idx = int(np.round(distance_aorta_lcca / bct_resolution, 0))
        lcca, _ = left_common_carotid_II(
            bct.coordinates[lcca_idx], resolution=1, rng=rng
        )

        distance_bct_lsca = normal(41, 2.5)
        idx += int(np.round(distance_bct_lsca / aorta_resolution, 0))
        lsa, _ = left_subclavian(aorta.coordinates[-idx], 1, rng)
        return [aorta, bct, rcca, rsa, lcca, lsa]

    def _create_type_I(self, rng, normal, aorta_resolution, aorta):
        distance_aorta_end_bct = normal(36, 5)
        idx = int(np.round(distance_aorta_end_bct / aorta_resolution, 0))
        bct, bct_chs_points = brachiocephalic_trunk_static(
            aorta.coordinates[-idx], 1, rng
        )

        rsa, _ = right_subclavian(bct.coordinates[-1], bct_chs_points[-1], 1, rng)
        rcca, _ = right_common_carotid(bct.coordinates[-1], 1, rng)

        distance_bct_lcca = normal(16, 4)
        idx += int(np.round(distance_bct_lcca / aorta_resolution, 0))
        lcca, _ = left_common_carotid(aorta.coordinates[-idx], 1, rng)

        distance_lcca_lsca = normal(28, 3)
        idx += int(np.round(distance_lcca_lsca / aorta_resolution, 0))
        lsa, _ = left_subclavian(aorta.coordinates[-idx], 1, rng)
        return [aorta, bct, rcca, rsa, lcca, lsa]
