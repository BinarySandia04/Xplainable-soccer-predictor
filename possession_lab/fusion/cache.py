"""npz persistence for a tallied poll.

The archive is version-1 additive: ballots are arrays (``src_<key>_*``), everything
else — weights, roles, diagnostics — is JSON metadata.  Older archives that lack the
role/diagnostics fields load unchanged: roles are re-derived from the voter registry
and diagnostics simply start empty.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .ballot import FusionResult, SourceVote, VoterRole
from .geometry import PRIOR
from .voters import SOURCE_LABELS, role_for_key


def save_fusion(path: str | Path, result: FusionResult) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "version": 1,
        "start": result.start,
        "stop": result.stop,
        "weights": result.meta.get("weights", {}),
        "combine_mode": result.meta.get("combine_mode", "log"),
        "prior": PRIOR.tolist(),
        "perception": bool(result.meta.get("perception", False)),
        "baseline": result.baseline_codes is not None,
        "sources": [
            {
                "key": key,
                "label": source.label,
                "role": VoterRole(source.role).value,
                "diagnostics": {str(k): float(v) for k, v in source.diagnostics.items()},
                "details": sorted(source.details),
            }
            for key, source in result.sources.items()
        ],
    }
    np.savez_compressed(path, **result.to_arrays(), meta=np.asarray(json.dumps(meta)))
    path.with_suffix(path.suffix + ".json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path


def load_fusion(path: str | Path) -> FusionResult | None:
    path = Path(path)
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(str(archive["meta"]))
        arrays = {name: archive[name] for name in archive.files if name != "meta"}
    sources: dict[str, SourceVote] = {}
    for item in meta.get("sources", []):
        key = str(item["key"])
        details = {
            str(detail): arrays[f"src_{key}_{detail}"]
            for detail in item.get("details", [])
            if f"src_{key}_{detail}" in arrays
        }
        role = VoterRole(item["role"]) if "role" in item else role_for_key(key)
        diagnostics = {str(k): float(v) for k, v in (item.get("diagnostics") or {}).items()}
        sources[key] = SourceVote(
            key=key,
            label=str(item.get("label", SOURCE_LABELS.get(key, key))),
            role=role,
            probability=np.asarray(arrays[f"src_{key}_probability"], dtype=np.float64),
            weight=np.asarray(arrays[f"src_{key}_weight"], dtype=np.float64),
            active=np.asarray(arrays[f"src_{key}_active"], dtype=bool),
            status=np.asarray(arrays[f"src_{key}_status"], dtype=np.int8),
            ball=np.asarray(arrays[f"src_{key}_ball"], dtype=np.float64),
            details=details,
            diagnostics=diagnostics,
        )
    codes = np.asarray(arrays["codes"], dtype=np.int8)
    baseline_codes = (
        np.asarray(arrays["baseline_codes"], dtype=np.int8)
        if meta.get("baseline", True)
        else None
    )
    return FusionResult(
        start=int(meta.get("start", 0)),
        stop=int(meta.get("stop", codes.size)),
        codes=codes,
        fused_probability=np.asarray(arrays["fused_probability"], dtype=np.float64),
        posterior=np.asarray(arrays["posterior"], dtype=np.float64),
        confidence=np.asarray(arrays["confidence"], dtype=np.float64),
        entropy=np.asarray(arrays["entropy"], dtype=np.float64),
        consensus_ball=np.asarray(arrays["consensus_ball"], dtype=np.float64),
        consensus_confidence=np.asarray(arrays["consensus_confidence"], dtype=np.float64),
        event_dead=np.asarray(arrays["event_dead"], dtype=bool),
        sources=sources,
        baseline_codes=baseline_codes,
        meta={
            "weights": meta.get("weights", {}),
            "combine_mode": meta.get("combine_mode", "log"),
            "perception": meta.get("perception", False),
        },
    )
