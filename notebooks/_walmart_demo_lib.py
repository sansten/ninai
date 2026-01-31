
import re
from dataclasses import dataclass


def normalize_feature(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


@dataclass
class Pref:
    must_have_features: list[str]
    avoid_features: list[str]


def feature_match_score(pref: Pref, item_features: list[str]) -> float:
    feats = {normalize_feature(f) for f in (item_features or [])}
    must = {normalize_feature(f) for f in pref.must_have_features}
    avoid = {normalize_feature(f) for f in pref.avoid_features}
    base = (len(must & feats) / max(1, len(must))) if must else 0.0
    penalty = (len(avoid & feats) / max(1, len(avoid))) if avoid else 0.0
    return max(0.0, base - penalty)
