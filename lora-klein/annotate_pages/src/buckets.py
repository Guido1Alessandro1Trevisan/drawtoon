"""FLUX.2 Klein 9-bucket grid. Mirrored from
~/Desktop/drawtoon-next/drawtoon/backend/generate/generate.py — keep in sync.
"""

from __future__ import annotations


# (name, (W, H))
BUCKETS: dict[str, tuple[int, int]] = {
    "tall-splash":    (704, 1408),  # 1:2
    "manga-page":     (768, 1344),  # 4:7
    "portrait":       (832, 1216),  # ~2:3
    "soft-portrait":  (896, 1152),  # ~4:5
    "square":        (1024, 1024),  # 1:1
    "soft-landscape":(1152,  896),  # ~5:4
    "landscape":     (1216,  832),  # ~3:2
    "wide":          (1344,  768),  # 7:4
    "cinematic":     (1408,  704),  # 2:1
}

_BUCKET_ASPECTS: dict[str, float] = {
    name: w / h for name, (w, h) in BUCKETS.items()
}


def closest_bucket(w_px: int, h_px: int) -> str:
    """Map an arbitrary (w, h) to the nearest named generation bucket by aspect."""
    h = max(1, int(h_px))
    ar = max(1, int(w_px)) / h
    return min(_BUCKET_ASPECTS, key=lambda name: abs(_BUCKET_ASPECTS[name] - ar))
