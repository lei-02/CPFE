"""PFEM Transolver backbone and CPFE prompt construction."""

from .features import FeatureLayout, build_node_features, enforce_displacement_bc
from .transolver import PhysicsAttention, TransolverCPFE

__all__ = [
    "FeatureLayout",
    "PhysicsAttention",
    "TransolverCPFE",
    "build_node_features",
    "enforce_displacement_bc",
]
