"""The episode ledger: which seeds were issued, accepted, or rejected and why.

WHY THIS EXISTS
---------------
Collection is 150 episodes across weeks and several sessions, drawn from a
PRE-PARTITIONED seed stream (CLAUDE.md section 8, dataset design). Without a
persistent record, session three has no idea what sessions one and two consumed,
a re-recorded failure is indistinguishable from a second copy of the same spawn,
and the held-out leak check cannot be PROVEN afterwards - only asserted.

CRASH-SAFE BY SHAPE, NOT BY CARE
--------------------------------
Append-only JSON Lines, flushed and `fsync`ed per line, and state is rebuilt by
replaying the file. There is no in-place update to interrupt: a crash mid-session
can lose at most the line being written, and a half-written final line is skipped
on load rather than poisoning the ledger. That is also why the ledger is not a
pickle or a single JSON document - both rewrite the whole file on every change,
which is exactly the operation a crash corrupts.

The ledger records decisions, never data: the episode itself is the .npz, and
`accept` stores the path to it.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
from typing import Dict, Iterable, List, Optional

EVENTS = ("session", "issue", "accept", "discard", "error")


class EpisodeLedger:
    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    # ---- writing -------------------------------------------------------
    def append(self, event: str, **fields) -> dict:
        if event not in EVENTS:
            raise ValueError("unknown ledger event %r (expected one of %s)"
                             % (event, ", ".join(EVENTS)))
        row = dict(t=_dt.datetime.now().isoformat(timespec="seconds"),
                   event=event, **fields)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=str) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        return row

    # ---- reading -------------------------------------------------------
    def rows(self) -> List[dict]:
        if not os.path.exists(self.path):
            return []
        out = []
        with open(self.path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    # A torn final line is what a crash mid-write looks like.
                    # Skipping it is correct; rewriting the file is not.
                    continue
        return out

    def state(self) -> Dict[int, dict]:
        """seed -> its LAST decision. A discarded seed is re-issuable, which is
        what proposal 3.3.8's re-recording of failures means in practice."""
        st: Dict[int, dict] = {}
        for r in self.rows():
            if r["event"] in ("accept", "discard", "error") and "seed" in r:
                st[int(r["seed"])] = r
        return st

    def accepted(self) -> Dict[int, dict]:
        return {s: r for s, r in self.state().items() if r["event"] == "accept"}

    def summary(self) -> dict:
        rows = self.rows()
        st = self.state()
        return dict(
            file=self.path, lines=len(rows),
            sessions=sum(1 for r in rows if r["event"] == "session"),
            issued=len({int(r["seed"]) for r in rows
                        if r["event"] == "issue" and "seed" in r}),
            accepted=sum(1 for r in st.values() if r["event"] == "accept"),
            discarded=sum(1 for r in st.values() if r["event"] == "discard"),
            errors=sum(1 for r in st.values() if r["event"] == "error"),
            heldout_accepted=sum(1 for r in st.values()
                                 if r["event"] == "accept" and r.get("heldout")))

    def pending(self, stream: Iterable[int]) -> List[int]:
        """Seeds from the stream with no ACCEPT yet, in stream order."""
        done = set(self.accepted())
        return [int(s) for s in stream if int(s) not in done]


def seed_stream(path: Optional[str], count: int, start: int = 0) -> List[int]:
    """The pre-partitioned stream, or a contiguous fallback.

    A JSON file of seeds is the intended input: pre-partitioning BEFORE
    collection is what makes held-out leakage provable rather than audited. The
    fallback exists so the recorder is runnable today, and says so.
    """
    if path:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        seeds = data["seeds"] if isinstance(data, dict) else data
        return [int(s) for s in seeds]
    return list(range(start, start + count))
