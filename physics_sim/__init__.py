"""Physics homework helpers, split from the notebook."""

from .common import BoxShape, RigidBody, RigidBodyState, quat_identity, vec3
from .part1 import build_part1_render_data, simulate_part1
from .part2 import build_part2_render_data, simulate_part2
from .part3 import (
    BruteForceBroadPhaseFixed,
    Part3Case,
    Part3Solver,
    build_part3_render_data,
    make_part3_box_pile_bodies,
    make_part3_solvers,
    simulate_part3,
    make_part3_many_box_solvers,
    simulate_part3_many_boxes,
    make_part3_many_box_bodies,
    make_many_box_grid_broadphase,
)
from .part4 import make_part4_mixed_box_bodies, make_part4_sap_broadphase, make_part4_lbvh_broadphase, simulate_part4, build_part4_render_data
from .rendering import render_sim_grid

__all__ = [
    "BoxShape",
    "RigidBody",
    "RigidBodyState",
    "quat_identity",
    "vec3",
    "build_part1_render_data",
    "simulate_part1",
    "build_part2_render_data",
    "simulate_part2",
    "BruteForceBroadPhaseFixed",
    "Part3Case",
    "Part3Solver",
    "build_part3_render_data",
    "make_part3_box_pile_bodies",
    "make_part3_solvers",
    "simulate_part3",
    "render_sim_grid",
    "make_part3_many_box_solvers",
    "simulate_part3_many_boxes",
    "make_part3_many_box_bodies",
    "make_many_box_grid_broadphase",
    "make_part4_mixed_box_bodies",
    "make_part4_sap_broadphase",
    "make_part4_lbvh_broadphase",
    "simulate_part4",
    "build_part4_render_data",
]
