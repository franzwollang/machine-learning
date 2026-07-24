"""Evidence gate contracts (SI S3.4, S3.5, S3.6)."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


class EditType(Enum):
    SPLIT = "split"
    PRUNE = "prune"
    MERGE = "merge"
    WARP = "warp"


@dataclass
class EditProposal:
    """A candidate structural edit before evidence scoring (SI S3.4)."""
    edit_type: EditType
    affected_node_ids: list[int]
    diagnostic_strength: float       # priority-queue key


@dataclass
class EvidenceRegion:
    """The localized region scored by the evidence gate (SI S3.4)."""
    core_node_ids: list[int]         # V_core
    ring_node_ids: list[int]         # neighbor ring
    transition_counts: np.ndarray    # (|V_aff|, max_J) counts n_{i->j}


@dataclass
class EvidenceVerdict:
    """Result of evidence scoring for a single edit (SI S3.4)."""
    accepted: bool
    f_dm_edit: float                 # F_DM(R; M_edit)
    f_dm_keep: float                 # F_DM(R; M_keep)
    log_bayes_factor: float          # f_dm_keep - f_dm_edit
    margin: float                    # log(tau_BF) threshold used
    proposal: EditProposal
