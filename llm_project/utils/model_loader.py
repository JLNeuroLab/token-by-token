from __future__ import annotations
import os
import json
from pathlib import Path
from typing import Optional, List, Dict, Any


def _registry_paths(family: str, root: str = "."):
    base = Path(root) / "experiments" / "models" / family
    return base / "registry.json", base / "saved_models"


def _load_registry(family: str, root: str = ".") -> List[Dict[str, Any]]:
    reg_path, _ = _registry_paths(family, root)
    if not reg_path.exists():
        return []
    try:
        return json.loads(reg_path.read_text(encoding="utf-8"))
    except Exception:
        return []


def resolve_final_ckpt(
    family: str,
    # --ckpt/--sm/--saved_model: best|latest|@0|0|run_id|shortid|path
    selector: Optional[str],
    sm_id: Optional[str],  # --sm_id: run_id or short-id suffix
    sm_idx: Optional[int],  # --sm_idx: registry index (sorted by metric)
    root: str = ".",
    sort_metric: str = "perplexity",  # choose the metric used to rank "best"
) -> Optional[str]:
    """
    Return a path to the requested final checkpoint, or None if not found.
    Works for any family that writes experiments/models/<family>/registry.json
    with entries holding: run_id, path, created_at/finished_at, and metrics.
    """
    if selector and (os.path.sep in selector or selector.endswith((".pth", ".pkl"))):
        return selector if os.path.exists(selector) else None

    reg = _load_registry(family, root)
    if not reg:
        return None

    # Rank by metric (lower==better). If missing, use +inf.
    def metric_val(r):
        v = r.get(sort_metric, None)
        try:
            return float(v)
        except Exception:
            return float("inf")

    rows_by_metric = sorted(reg, key=metric_val)  # best first
    rows_by_time = sorted(
        reg,
        key=lambda r: (r.get("finished_at") or r.get("created_at") or ""),
        reverse=True,
    )

    def path_of(row):
        return row.get("path") or ""

    # 1) explicit numeric index
    if sm_idx is not None and 0 <= sm_idx < len(rows_by_metric):
        return path_of(rows_by_metric[sm_idx])

    sel = (selector or "").strip()

    # 2) symbolic selectors
    if sel.lower() == "best":
        return path_of(rows_by_metric[0])
    if sel.lower() in {"latest", "newest"}:
        return path_of(rows_by_time[0])

    # 3) @index or plain integer
    if sel.startswith("@"):
        try:
            i = int(sel[1:])
            if 0 <= i < len(rows_by_metric):
                return path_of(rows_by_metric[i])
        except ValueError:
            pass
    elif sel.isdigit():
        i = int(sel)
        if 0 <= i < len(rows_by_metric):
            return path_of(rows_by_metric[i])

    # 4) run_id / short-id
    def match_run_id(s):
        for r in reg:
            rid = r.get("run_id", "")
            if s == rid or s == Path(path_of(r)).stem:
                return path_of(r)
        return None

    def match_short(sfx):
        sfx = sfx.lower()
        cands = [r for r in reg if r.get("run_id", "").lower().endswith(sfx)]
        return path_of(cands[0]) if len(cands) == 1 else None

    if sm_id:
        p = match_run_id(sm_id) or match_short(sm_id)
        if p:
            return p
    if sel:
        p = match_run_id(sel) or match_short(sel)
        if p:
            return p

    # 5) last chance: treat selector as a file under saved_models/
    _, saved_dir = _registry_paths(family, root)
    maybe = saved_dir / sel
    if sel and maybe.exists():
        return str(maybe)

    return None
