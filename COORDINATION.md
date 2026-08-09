# Swarm COORDINATION protocol (credit burn)

Operational rules live in `.cursor/rules/swarm-coordination.mdc`. This file is the
**schema stub** for `COORDINATION.jsonl` mailboxes (one per `coord/A*` branch).

**Durable branches:** `coord/A1` … `coord/A6` and `coord/integration`. Do not rename
branches when issues change — reassign work in `SWARM_TASKS.json` only.

## File roles

| File | Writer | Role |
|---|---|---|
| `COORDINATION.jsonl` (per agent branch) | that agent only | Append-only mailbox |
| `SWARM_TASKS.json` | **A1 only** | Pre-buffered task queues (2–3 deep per agent) |
| `coord/integration` | **A1 only** (merges) | Integration branch; workers pull, never push here |

## Line schema (`COORDINATION.jsonl`)

One JSON object per line. Unknown fields are allowed; keep lines compact.

```json
{
  "ts": "2026-08-09T12:00:00-05:00",
  "from": "A2",
  "type": "CLAIM",
  "issue": 44,
  "task_id": "A2-T1",
  "files": ["code/proteus/src/proteus/stage1/recursion.py"],
  "note": "optional one-liner"
}
```

### Required fields

- `ts` — ISO 8601 timestamp
- `from` — `A1` … `A6`
- `type` — see rule file (`HELLO`, `CLAIM`, `DONE`, `BLOCKED`, `NOTE`,
  `REQUEST_TRACKER`, `TASK_ASSIGN`, `MERGE_START`, `MERGE_DONE`, `TASK_REFILL`)

### Recommended fields

- `issue` — OPEN_ISSUES number when applicable
- `task_id` — id from `SWARM_TASKS.json`
- `files` — paths you claim or touched
- `note` — one line
- `merge_order` — on `MERGE_START`/`MERGE_DONE`, optional tip SHAs

## Reading siblings

```bash
git fetch origin
for b in coord/A1 coord/A2 coord/A3 coord/A4 coord/A5 coord/A6 coord/integration; do
  git show "origin/${b}:COORDINATION.jsonl" 2>/dev/null | tail -n 15 \
    > "/tmp/coord_${b##*/}.jsonl" || true
done
git show origin/coord/integration:SWARM_TASKS.json > /tmp/SWARM_TASKS.json
```

## MERGE_START / MERGE_DONE (worker duty)

On seeing **`MERGE_START`** from A1: finish current edit → commit → push your branch →
**stop new commits** until **`MERGE_DONE`**. Next turn: merge `origin/coord/integration`
into your branch before further work.
