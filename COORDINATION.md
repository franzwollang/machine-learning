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
  `REQUEST_TRACKER`, `TASK_ASSIGN`, `MERGE_DONE`, `TASK_REFILL`)

### Recommended fields

- `issue` — OPEN_ISSUES number when applicable
- `task_id` — id from `SWARM_TASKS.json`
- `files` — paths you claim or touched
- `note` — one line
- `integration_sha` — on `MERGE_DONE`: SHA of `coord/integration` after the merge
- `integrated_sha` — optional worker `NOTE` after merging integration into their branch

## Reading siblings

```bash
git fetch origin
for b in coord/A1 coord/A2 coord/A3 coord/A4 coord/A5 coord/A6 coord/integration; do
  git show "origin/${b}:COORDINATION.jsonl" 2>/dev/null | tail -n 15 \
    > "/tmp/coord_${b##*/}.jsonl" || true
done
git show origin/coord/integration:SWARM_TASKS.json > /tmp/SWARM_TASKS.json
```

## Integration (turn-boundary only — ~20 min agent cadence)

Agents **cannot** react mid-turn to A1 merging. There is no `MERGE_START` pause.

**A1:** when ready, merge current worker tips into `coord/integration` (best-effort
snapshot), test, append **`MERGE_DONE`** with `integration_sha`, push. Tips that land
during the merge are scooped up next cycle.

**Workers (start of every turn, before new work):**

1. `git fetch origin`
2. If `origin/coord/integration` is ahead of / not in your branch:  
   `git merge origin/coord/integration` (resolve if needed), commit
3. Then work and push normally — do **not** wait or pause for A1

You only need to know that A1 **already merged** since your last turn (new
`MERGE_DONE` / newer integration tip), not that a merge is happening now.
