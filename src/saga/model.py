"""Pairwise moment estimation of global and query-dependent source alignment."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

MARGIN = 1e-6


def group_folds(groups, count=3, seed=42):
    unique = np.unique(groups)
    if len(unique) < count:
        raise ValueError(f"At least {count} independent groups are required.")
    np.random.default_rng(seed).shuffle(unique)
    assignments = {group: i % count for i, group in enumerate(unique)}
    return np.asarray([assignments[group] for group in groups])


@dataclass
class Projection:
    mean: np.ndarray
    basis: np.ndarray
    scale: float

    @classmethod
    def fit(cls, queries, dimension=64):
        mean = queries.mean(0)
        _, _, directions = np.linalg.svd(queries - mean, full_matrices=False)
        basis = directions[: min(dimension, len(queries) - 1)].T
        features = (queries - mean) @ basis
        scale = max(float(np.sqrt((features * features).sum(1).mean())), 1e-12)
        return cls(mean, basis, scale)

    def transform(self, queries):
        return (queries - self.mean) @ self.basis / self.scale


def observed_pairs(gram, valid, sources):
    upper = np.triu(np.ones(gram.shape[1:], dtype=bool), 1)
    i, a, b = np.where(valid[:, :, None] & valid[:, None, :] & upper[None])
    if not len(i):
        raise ValueError("Alignment estimation needs co-observed source pairs.")
    return i, sources[i, a], sources[i, b], gram[i, a, b]


def predict_alignment(coefficients, features):
    return MARGIN + (1 - 2 * MARGIN) * expit(
        coefficients[0] + features @ coefficients[1:]
    )


def pair_loss(parameters, features, pairs, regularization, source_count):
    coefficients = parameters.reshape(features.shape[1] + 1, source_count)
    raw = expit(coefficients[0] + features @ coefficients[1:])
    alignment = MARGIN + (1 - 2 * MARGIN) * raw
    i, a, b, target = pairs
    residual = alignment[i, a] * alignment[i, b] - target
    loss = np.mean(residual**2) + regularization * np.sum(coefficients[1:] ** 2)
    gradient = np.zeros_like(alignment)
    factor = 2 * residual / len(target)
    np.add.at(gradient, (i, a), factor * alignment[i, b])
    np.add.at(gradient, (i, b), factor * alignment[i, a])
    gradient *= (1 - 2 * MARGIN) * raw * (1 - raw)
    derivative = np.vstack(
        [gradient.sum(0), features.T @ gradient + 2 * regularization * coefficients[1:]]
    )
    return float(loss), derivative.ravel()


def dense_pair_loss(parameters, features, target, regularization, source_count):
    """Vectorized three-source kernel for complete response panels."""
    n, dimension = features.shape
    coefficients = parameters.reshape(dimension + 1, source_count)
    raw = expit(coefficients[0] + features @ coefficients[1:])
    alignment = MARGIN + (1 - 2 * MARGIN) * raw
    pairs = [(0, 1), (0, 2), (1, 2)]
    predicted = np.stack([alignment[:, a] * alignment[:, b] for a, b in pairs], 1)
    residual = predicted - target
    loss = np.mean(residual**2) + regularization * np.sum(coefficients[1:] ** 2)
    gradient = np.zeros_like(alignment)
    for j, (a, b) in enumerate(pairs):
        gradient[:, a] += 2 * residual[:, j] * alignment[:, b] / (n * 3)
        gradient[:, b] += 2 * residual[:, j] * alignment[:, a] / (n * 3)
    gradient *= (1 - 2 * MARGIN) * raw * (1 - raw)
    derivative = np.vstack(
        [gradient.sum(0), features.T @ gradient + 2 * regularization * coefficients[1:]]
    )
    return float(loss), derivative.ravel()


def fit_coefficients(features, target, regularization, source_count, dense=False):
    loss = dense_pair_loss if dense else pair_loss
    fits = []
    for seed in (42, 43, 44):
        initial = np.random.default_rng(seed).normal(
            0, 0.02, (features.shape[1] + 1, source_count)
        )
        initial[0] = np.log(4)
        fit = minimize(
            loss,
            initial.ravel(),
            args=(features, target, regularization, source_count),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8},
        )
        fits.append(fit)
    best = min(fits, key=lambda result: result.fun)
    if not best.success:
        raise RuntimeError(f"Alignment fit did not converge: {best.message}")
    return best.x.reshape(features.shape[1] + 1, source_count), {
        "objective": float(best.fun),
        "iterations": int(best.nit),
        "start_objectives": [float(fit.fun) for fit in fits],
    }


class AlignmentModel:
    """Estimate source alignments from unlabeled pairwise answer agreement."""

    def __init__(self, query=True, regularization=(1e-4, 1e-3, 1e-2), dimension=64):
        self.query = query
        self.regularization = tuple(regularization)
        self.dimension = dimension
        self.projection = None

    def fit(self, queries, gram, valid, sources, groups, source_count):
        q = np.asarray(queries, dtype=float)
        dense = (
            source_count == 3
            and valid.all()
            and np.array_equal(sources, np.tile(np.arange(3), (len(q), 1)))
        )

        def target(mask):
            if dense:
                return gram[mask][:, [0, 0, 1], [1, 2, 2]]
            return observed_pairs(gram[mask], valid[mask], sources[mask])

        selected = 0.0
        validation = None
        if self.query:
            folds = group_folds(groups)
            validation = np.zeros((len(self.regularization), 3))
            for fold in range(3):
                train, held = folds != fold, folds == fold
                projection = Projection.fit(q[train], self.dimension)
                x, test = projection.transform(q[train]), projection.transform(q[held])
                train_target, test_target = target(train), target(held)
                for j, penalty in enumerate(self.regularization):
                    coefficients, _ = fit_coefficients(
                        x, train_target, penalty, source_count, dense
                    )
                    s = predict_alignment(coefficients, test)
                    if dense:
                        prediction = np.stack(
                            [s[:, 0] * s[:, 1], s[:, 0] * s[:, 2], s[:, 1] * s[:, 2]], 1
                        )
                        validation[j, fold] = np.mean((prediction - test_target) ** 2)
                    else:
                        i, a, b, y = test_target
                        validation[j, fold] = np.mean((s[i, a] * s[i, b] - y) ** 2)
            selected = self.regularization[int(validation.mean(1).argmin())]
            self.projection = Projection.fit(q, self.dimension)
            features = self.projection.transform(q)
        else:
            features = np.empty((len(q), 0))
        self.coefficients, info = fit_coefficients(
            features, target(np.arange(len(q))), selected, source_count, dense
        )
        self.fit_info = {
            **info,
            "regularization": selected,
            "validation_loss": validation.tolist() if validation is not None else None,
        }
        return self

    def predict(self, queries):
        features = (
            self.projection.transform(queries)
            if self.query
            else np.empty((len(queries), 0))
        )
        return predict_alignment(self.coefficients, features)

    def save(self, path):
        arrays = {"coefficients": self.coefficients, "query": np.array(self.query)}
        if self.projection is not None:
            arrays.update(
                mean=self.projection.mean,
                basis=self.projection.basis,
                scale=self.projection.scale,
            )
        np.savez_compressed(path, **arrays)
