"""Risk minimization and extractive answer selection."""

from itertools import combinations

import numpy as np


def simplex_weights(alignment):
    """Minimize (1 - p @ s)^2 + sum(p^2 * (1 - s^2)) on the simplex."""
    s = np.asarray(alignment, dtype=float)
    if s.ndim != 1 or not len(s) or not np.all((s > 0) & (s < 1)):
        raise ValueError("Alignment must be a nonempty vector with values in (0, 1).")
    size = len(s)
    quadratic = np.outer(s, s) + np.diag(1 - s**2)
    candidates = []
    for count in range(1, size + 1):
        for support in combinations(range(size), count):
            indices = list(support)
            submatrix = quadratic[np.ix_(indices, indices)]
            a = np.linalg.solve(submatrix, s[indices])
            b = np.linalg.solve(submatrix, np.ones(count))
            active = a + b * (1 - a.sum()) / b.sum()
            if active.min() >= -1e-10:
                weights = np.zeros(size)
                weights[indices] = np.maximum(active, 0)
                weights /= weights.sum()
                risk = 1 - 2 * weights @ s + weights @ quadratic @ weights
                candidates.append((float(risk), weights))
    return min(candidates, key=lambda pair: pair[0])[1]


def risk_weights(alignment, valid, source_indices):
    """Map persistent source qualities to each instance's available candidates."""
    local = np.take_along_axis(alignment, source_indices, axis=1)
    weights = np.zeros_like(local)
    for i in range(len(local)):
        available = np.flatnonzero(valid[i])
        if not len(available):
            raise ValueError("Every instance must have an available candidate.")
        weights[i, available] = simplex_weights(local[i, available])
    return weights


def select_answer(gram, weights, valid):
    """Choose the available candidate closest in cosine to the weighted center.

    The center norm is common to all candidates. Ties use the first slot.
    """
    weights = np.where(valid, weights, 0)
    shares = weights / np.maximum(weights.sum(1, keepdims=True), 1e-30)
    support = np.einsum("njl,nl->nj", gram, shares)
    support /= np.sqrt(np.maximum(np.diagonal(gram, axis1=1, axis2=2), 1e-30))
    support[~valid] = -np.inf
    return support.argmax(1)
