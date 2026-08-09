"""Stage 2 complex construction, diagnostics, and density reconstruction."""

from .dual_flow import (
    ConservativeBPResult,
    DualAdjacencyDict,
    DualDryRunResult,
    DualFlowConfig,
    affected_subgraph_connected,
    build_dual_adjacency,
    build_dual_adjacency_from_complex,
    dry_run_dual_from_edit,
    resolve_dual_connected,
    solve_conservative_pressures,
)
from .flag_complex import (
    FlagComplexConfig,
    FlagComplexResult,
    build_flag_complex,
    flag_complex_from_scaffold,
    simplex_volume,
)

__all__ = [
    "ConservativeBPResult",
    "DualAdjacencyDict",
    "DualDryRunResult",
    "DualFlowConfig",
    "FlagComplexConfig",
    "FlagComplexResult",
    "affected_subgraph_connected",
    "build_dual_adjacency",
    "build_dual_adjacency_from_complex",
    "build_flag_complex",
    "dry_run_dual_from_edit",
    "flag_complex_from_scaffold",
    "resolve_dual_connected",
    "simplex_volume",
    "solve_conservative_pressures",
]
