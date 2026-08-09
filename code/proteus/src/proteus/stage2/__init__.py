"""Stage 2 complex construction, diagnostics, and density reconstruction."""

from .dual_flow import (
    DualAdjacencyDict,
    DualFlowConfig,
    affected_subgraph_connected,
    build_dual_adjacency,
    build_dual_adjacency_from_complex,
    resolve_dual_connected,
)
from .flag_complex import (
    FlagComplexConfig,
    FlagComplexResult,
    build_flag_complex,
    flag_complex_from_scaffold,
    simplex_volume,
)

__all__ = [
    "DualAdjacencyDict",
    "DualFlowConfig",
    "FlagComplexConfig",
    "FlagComplexResult",
    "affected_subgraph_connected",
    "build_dual_adjacency",
    "build_dual_adjacency_from_complex",
    "build_flag_complex",
    "flag_complex_from_scaffold",
    "resolve_dual_connected",
    "simplex_volume",
]
