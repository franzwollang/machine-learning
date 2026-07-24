"""Stage 2 complex construction, diagnostics, and density reconstruction."""

from .flag_complex import (
    FlagComplexConfig,
    FlagComplexResult,
    build_flag_complex,
    flag_complex_from_scaffold,
    simplex_volume,
)

__all__ = [
    "FlagComplexConfig",
    "FlagComplexResult",
    "build_flag_complex",
    "flag_complex_from_scaffold",
    "simplex_volume",
]
