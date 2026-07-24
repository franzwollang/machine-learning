"""Evidence scoring and structure-selection gates (SI S3.4-S3.6, S10.4)."""

from proteus.evidence.dm_score import (
    NodeTransition,
    bdeu_alpha,
    evaluate_edit,
    f_dm,
    node_log_marginal,
)
from proteus.evidence.gate import (
    EvidenceGate,
    GateConfig,
    edit_budget,
    gate_window,
    hysteresis_window,
    score_edit,
)
from proteus.evidence.star_matrix import (
    RHO_MIN_CONSERVATIVE,
    RHO_MIN_DEFAULT,
    condition_ratio,
    is_evidence_bearing,
    quarantined_nodes,
    star_incidence_matrix,
)

__all__ = [
    "NodeTransition",
    "bdeu_alpha",
    "node_log_marginal",
    "f_dm",
    "evaluate_edit",
    "star_incidence_matrix",
    "condition_ratio",
    "is_evidence_bearing",
    "quarantined_nodes",
    "RHO_MIN_DEFAULT",
    "RHO_MIN_CONSERVATIVE",
    "GateConfig",
    "gate_window",
    "hysteresis_window",
    "edit_budget",
    "score_edit",
    "EvidenceGate",
]
