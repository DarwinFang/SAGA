import numpy as np
from scipy.optimize import check_grad, minimize

from saga.decision import risk_weights, select_answer, simplex_weights
from saga.model import (
    dense_pair_loss,
    fit_coefficients,
    group_folds,
    observed_pairs,
    pair_loss,
    predict_alignment,
)


def test_pair_gradient_and_dense_equivalence():
    rng = np.random.default_rng(42)
    features = rng.normal(size=(11, 4))
    answers = rng.normal(size=(11, 3, 7))
    answers /= np.linalg.norm(answers, axis=2, keepdims=True)
    gram = answers @ answers.swapaxes(1, 2)
    valid = np.ones((11, 3), dtype=bool)
    source = np.tile(np.arange(3), (11, 1))
    pairs = observed_pairs(gram, valid, source)
    parameters = rng.normal(size=15)
    args = (features, pairs, 0.01, 3)
    error = check_grad(
        lambda value: pair_loss(value, *args)[0],
        lambda value: pair_loss(value, *args)[1],
        parameters,
    )
    assert error < 1e-6
    a, da = pair_loss(parameters, *args)
    b, db = dense_pair_loss(
        parameters, features, gram[:, [0, 0, 1], [1, 2, 2]], 0.01, 3
    )
    np.testing.assert_allclose(a, b, atol=1e-14)
    np.testing.assert_allclose(da, db, atol=1e-14)


def test_risk_solution_against_independent_optimizer():
    rng = np.random.default_rng(8)
    for size in (2, 3, 5):
        for s in rng.uniform(0.02, 0.98, (10, size)):

            def risk(p):
                return (1 - p @ s) ** 2 + np.sum(p * p * (1 - s * s))

            weights = simplex_weights(s)
            fit = minimize(
                risk,
                np.ones(size) / size,
                method="SLSQP",
                bounds=[(0, 1)] * size,
                constraints=[{"type": "eq", "fun": lambda p: p.sum() - 1}],
                options={"ftol": 1e-12},
            )
            assert fit.success
            assert abs(risk(weights) - fit.fun) < 1e-8
            assert np.all(weights >= 0)
            np.testing.assert_allclose(weights.sum(), 1)


def test_sparse_source_identity_and_single_candidate():
    s = np.array([[0.9, 0.8, 0.5], [0.9, 0.8, 0.5]])
    sources = np.array([[2, 0], [1, 0]])
    valid = np.array([[True, True], [True, False]])
    weights = risk_weights(s, valid, sources)
    np.testing.assert_allclose(weights[0], simplex_weights(np.array([0.5, 0.9])))
    np.testing.assert_array_equal(weights[1], [1, 0])
    gram = np.broadcast_to(np.eye(2), (2, 2, 2)).copy()
    np.testing.assert_array_equal(select_answer(gram, weights, valid), [1, 0])


def test_groups_never_cross_validation_folds():
    groups = np.repeat(np.arange(9), [1, 2, 1, 3, 2, 1, 1, 2, 2])
    folds = group_folds(groups)
    for group in np.unique(groups):
        assert len(np.unique(folds[groups == group])) == 1
    assert set(folds) == {0, 1, 2}


def test_population_moments_identify_alignment():
    truth = np.array([0.9, 0.8, 0.5])
    target = np.tile(
        [truth[0] * truth[1], truth[0] * truth[2], truth[1] * truth[2]], (20, 1)
    )
    coefficients, _ = fit_coefficients(np.empty((20, 0)), target, 0, 3, dense=True)
    predicted = predict_alignment(coefficients, np.empty((1, 0)))[0]
    np.testing.assert_allclose(predicted, truth, atol=1e-4)
