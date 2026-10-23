"""Decision rules and the paper's expertise-reversal construction."""

import hashlib
from itertools import combinations, permutations

import numpy as np

from .decision import select_answer
from .grouping import assign, cluster


def category_correlations(panel, quality, predictions, labels):
    """Within-category EM/count Pearson correlations with exact-text attribution."""
    categories = sorted(set(labels))
    masks = [labels == label for label in categories]
    em = np.array([quality[mask].mean(0) for mask in masks])
    result = {"labels": categories, "source_em": em.tolist(), "methods": {}}
    for name, picks in predictions.items():
        chosen = panel.texts[np.arange(len(panel)), picks]
        matches = (panel.texts == chosen[:, None]) & panel.valid
        shares = matches / matches.sum(1, keepdims=True)
        counts = np.array([shares[mask].sum(0) for mask in masks])
        correlation = []
        for accuracy, count in zip(em, counts):
            if np.ptp(accuracy) == 0 or np.ptp(count) < 1e-12:
                raise ValueError("Pearson correlation is undefined for a constant row")
            correlation.append(float(np.corrcoef(accuracy, count)[0, 1]))
        result["methods"][name] = {"counts": counts.tolist(), "pearson": correlation,
                                   "mean_pearson": float(np.mean(correlation))}
    return result


def decision_ablations(panel, alignment):
    local = np.take_along_axis(alignment, panel.source_indices, axis=1)
    available = np.where(panel.valid, local, 0)
    normalized = available / available.sum(1, keepdims=True)
    return {
        "SAGA-HA": np.where(panel.valid, local, -np.inf).argmax(1),
        "SAGA-NC": select_answer(panel.gram, normalized, panel.valid),
    }


def primary_reversal(panel, quality, region_count=4, minimum=12, seed=42):
    regions = assign(
        panel.queries.astype(float), cluster(panel.queries.astype(float), region_count)
    )
    eligible = []
    for left, right in combinations(range(panel.valid.shape[1]), 2):
        delta = quality[:, left] - quality[:, right]
        for ra, rb in permutations(range(region_count), 2):
            first = np.flatnonzero((regions == ra) & (delta > 1e-8))
            second = np.flatnonzero((regions == rb) & (delta < -1e-8))
            half = min(len(first), len(second))
            if half >= minimum:
                eligible.append((2 * half, left, right, ra, rb, first, second))
    if not eligible:
        raise ValueError("No balanced reversal batch satisfies the minimum size")
    n, left, right, ra, rb, first, second = min(
        eligible, key=lambda x: (-x[0], *x[1:5])
    )

    def order(indices):
        return sorted(
            map(int, indices),
            key=lambda i: (
                hashlib.sha256(
                    (str(seed) + "|" + str(panel.ids[i])).encode()
                ).hexdigest(),
                str(panel.ids[i]),
            ),
        )

    indices = sorted(order(first)[: n // 2] + order(second)[: n // 2])
    return {
        "n": n,
        "source_pair": [left, right],
        "region_pair": [ra, rb],
        "indices": indices,
        "ids": panel.ids[indices].tolist(),
        "regions": regions.tolist(),
    }
