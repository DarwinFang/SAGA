"""Source reliability and semantic-agreement baselines.

Candidates occupy row-local slots. Optional source indices connect those slots
to persistent source identities across instances.
"""

from collections import Counter

import numpy as np
from scipy.stats import chi2


def choose(gram, weights, valid):
    shares = weights / np.maximum(weights.sum(1, keepdims=True), 1e-30)
    support = np.einsum("njl,nl->nj", gram, shares)
    support /= np.sqrt(np.maximum(np.diagonal(gram, axis1=1, axis2=2), 1e-30))
    support[~valid] = -np.inf
    picks = support.argmax(1)
    picks[~valid.any(1)] = -1
    return picks, support, shares


def _fit(gram, valid, gamma=None, source_indices=None, n_sources=None):
    n, width = valid.shape
    compact = source_indices is not None
    source = (
        np.broadcast_to(np.arange(width), valid.shape)
        if source_indices is None
        else np.asarray(source_indices, dtype=int)
    )
    size = width if n_sources is None else n_sources
    fitvalid = valid if gamma is None else valid & (valid.sum(1) > 1)[:, None]

    def totals(values):
        if not compact:
            return (values * fitvalid).sum(0)
        return np.bincount(source[fitvalid], weights=values[fitvalid], minlength=size)

    counts = totals(np.ones(valid.shape))
    prior = chi2.isf(0.025, np.maximum(counts, 1))
    prior[counts == 0] = 0
    beta = np.ones(size)
    previous, change = None, None
    for iteration in range(100):
        weights = fitvalid * beta[source]
        if gamma is not None:
            weights *= gamma
        _, _, shares = choose(gram, weights, fitvalid)
        projected = np.einsum("njl,nl->nj", gram, shares)
        center_norm = (shares * projected).sum(1)
        residual = (
            np.diagonal(gram, axis1=1, axis2=2) - 2 * projected + center_norm[:, None]
        )
        if gamma is not None or compact:
            residual = np.maximum(residual, 0)
        if gamma is not None:
            residual = residual / np.where(fitvalid, gamma, 1)
        beta = prior / np.maximum(totals(residual), 1e-5)
        if previous is not None:
            difference = shares - previous
            change = float(
                np.einsum("nj,njl,nl->", difference, gram, difference)
                / max(center_norm.sum(), 1e-30)
            )
            if change < 1e-9:
                break
        previous = shares.copy()
    if gamma is not None:
        singles = valid.sum(1) == 1
        shares[singles] = valid[singles]
    # Return the center computed before the final reliability update.
    return shares, {
        "beta": beta.tolist(),
        "observed_counts": counts.astype(int).tolist(),
        "iterations": iteration + 1,
        "change": change,
    }


def rasa_weights(gram, valid, source_indices=None, n_sources=None):
    return _fit(gram, valid, source_indices=source_indices, n_sources=n_sources)


def hrrasa_weights(gram, valid, gamma, source_indices=None, n_sources=None):
    return _fit(gram, valid, gamma, source_indices, n_sources)


def ngrams(sequence):
    return Counter(
        tuple(sequence[i : i + n])
        for n in range(1, 5)
        for i in range(max(len(sequence) - n + 1, 0))
    )


def gleu(hypothesis, reference):
    h, r = ngrams(hypothesis), ngrams(reference)
    denominator = max(sum(h.values()), sum(r.values()))
    return sum((h & r).values()) / denominator if denominator else 0.0


def local_agreement(gram, valid, texts):
    """Pinned HRRASA default: tokenize the first overlap argument only."""
    diag = np.diagonal(gram, axis1=1, axis2=2)
    distance = np.maximum(diag[:, :, None] + diag[:, None, :] - 2 * gram, 0)
    semantic = np.exp(
        -distance / np.maximum(diag[:, :, None] * diag[:, None, :], 1e-30)
    )
    gamma = np.ones(valid.shape)
    sequence = np.zeros(valid.shape)
    for i in range(len(gram)):
        observed = np.flatnonzero(valid[i])
        if len(observed) < 2:
            continue
        for j in observed:
            others = observed[observed != j]
            sequence[i, j] = np.mean(
                [gleu(texts[i][j].split(), texts[i][other]) for other in others]
            )
            gamma[i, j] = 0.5 * semantic[i, j, others].mean() + 0.5 * sequence[i, j]
    gamma[~valid] = 0
    return gamma, sequence

def bau_predictions(gram):
    """Best available source under empirical-prior mean disagreement (BAU)."""
    from itertools import combinations

    n, m, _ = gram.shape
    pairs = np.array(list(combinations(range(m), 2)))
    distances = np.clip(1 - gram[:, pairs[:, 0], pairs[:, 1]], 0, 2)
    distances += 0.1 - distances.min()
    values = [distances[:, np.any(pairs == k, axis=1)].ravel() for k in range(m)]
    mean = np.array([v.mean() for v in values])
    prior_var = mean.var(ddof=1)
    if prior_var == 0:
        errors = mean
    else:
        variance = np.array([np.mean((v - mu) ** 2) for v, mu in zip(values, mean)])
        variance += 0.1 * prior_var
        count = np.array([len(v) for v in values])
        errors = (mean.mean() / prior_var + count * mean / variance) / (
            1 / prior_var + count / variance
        )
    return np.full(n, errors.argmin(), dtype=int), errors


def aqer_predictions(answers, max_iterations=10000, tolerance=1e-6):
    """Iterative source quality and nearest weighted proxy selection (AQER)."""
    z = answers.astype(float)
    n, m, _ = z.shape
    gram = z @ z.swapaxes(1, 2)
    norms = np.linalg.norm(z, axis=2)
    weights = np.full(m, 1 / m)
    for iteration in range(max_iterations):
        previous = weights.copy()
        center_norm = np.sqrt(np.einsum("k,nkl,l->n", previous, gram, previous))
        if np.any(center_norm == 0):
            raise ValueError("AQER proxy has zero norm")
        similarities = np.einsum("nkl,l->nk", gram, previous) / center_norm[:, None] / norms
        grades = similarities.mean(0)
        span = np.ptp(grades)
        if span <= np.finfo(float).eps:
            raise ValueError("AQER source grades have zero normalization range")
        scaled = (grades - grades.min()) / span
        weights = scaled / scaled.sum()
        rmse = float(np.sqrt(np.mean((weights - previous) ** 2)))
        if rmse < tolerance:
            break
    center = np.einsum("k,nkd->nd", weights, z)
    similarities = np.einsum("nkd,nd->nk", z, center) / norms / np.linalg.norm(center, axis=1)[:, None]
    picks = np.isclose(similarities, similarities.max(1)[:, None], rtol=0, atol=1e-12).argmax(1)
    return picks, {"weights": weights.tolist(), "iterations": iteration + 1,
                   "rmse": rmse, "converged": rmse < tolerance}
