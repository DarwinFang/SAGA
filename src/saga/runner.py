"""Shared fitting and caching for the paper experiments."""

import hashlib
import json
from pathlib import Path

import numpy as np

from .baselines import aqer_predictions, bau_predictions, choose, hrrasa_weights, local_agreement, rasa_weights
from .data import write_json
from .decision import risk_weights, select_answer
from .model import AlignmentModel
from .vendor.smoothie import Smoothie

GLOBAL = "SAGA w/o Query"
QUERY = "SAGA"


def fingerprint(panel, config):
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
    for key in panel.__dataclass_fields__:
        value = getattr(panel, key)
        digest.update(str((key, value.dtype, value.shape)).encode())
        digest.update(value.tobytes())
    for path in sorted(Path(__file__).parent.rglob("*.py")):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def baseline_predictions(panel, supplementary=False):
    gram, valid, source = panel.gram, panel.valid, panel.source_indices
    predictions = {"SMS": choose(gram, valid, valid)[0]}
    reliability, _ = rasa_weights(gram, valid, source, len(panel.sources))
    predictions["RASA"] = choose(gram, reliability, valid)[0]
    if not valid.all():
        raise ValueError("The paper baselines require a complete source panel")
    predictions["BAU"] = bau_predictions(gram)[0]
    picks, info = aqer_predictions(panel.answers)
    if not info["converged"]:
        raise RuntimeError("AQER did not reach the configured convergence tolerance")
    predictions["AQER"] = picks
    if supplementary:
        gamma, _ = local_agreement(gram, valid, panel.texts.tolist())
        reliability, _ = hrrasa_weights(gram, valid, gamma, source, len(panel.sources))
        predictions["HRRASA"] = choose(gram, reliability, valid)[0]
        queries = panel.queries.astype(float)
        distance = ((queries * queries).sum(1)[:, None]
                    + (queries * queries).sum(1)[None] - 2 * queries @ queries.T)
        np.fill_diagonal(distance, np.inf)
        nearest = np.argsort(distance, axis=1, kind="stable")[:, 0]
        theta = []
        for neighbor in nearest:
            model = Smoothie(len(panel.sources), panel.answers.shape[-1])
            model.fit(panel.answers[neighbor:neighbor + 1])
            theta.append(model.theta.copy())
        predictions["Smoothie"] = np.asarray(theta).argmax(1)
    return predictions


def fit_panel(panel, directory, config, baselines=True, shared=False):
    """Fit the supplied unlabeled batch. Persist model parameters and predictions."""
    directory = Path(directory)
    signature = fingerprint(panel, {**config, "baselines": baselines, "shared": shared})
    cache = directory / "fit.json"
    if cache.exists() and json.loads(cache.read_text()).get("signature") == signature:
        with np.load(directory / "predictions.npz", allow_pickle=False) as arrays:
            return {key: arrays[key] for key in arrays.files}
    directory.mkdir(parents=True, exist_ok=True)
    gram = panel.gram
    predictions = baseline_predictions(panel) if baselines else {}
    arrays = {"ids": panel.ids}
    info = {}
    for name, conditional in ([(GLOBAL, False), (QUERY, True)] if shared else [(QUERY, True)]):
        model = AlignmentModel(
            query=conditional,
            regularization=config["regularization"],
            dimension=config["pca_dimension"],
        )
        model.fit(
            panel.queries,
            gram,
            panel.valid,
            panel.source_indices,
            panel.groups,
            len(panel.sources),
        )
        alignment = model.predict(panel.queries.astype(float))
        weights = risk_weights(alignment, panel.valid, panel.source_indices)
        predictions[name] = select_answer(gram, weights, panel.valid)
        arrays[name + "_alignment"] = alignment
        arrays[name + "_weights"] = weights
        info[name] = model.fit_info
        model.save(directory / ("saga.npz" if conditional else "shared_alignment.npz"))
    arrays.update(
        methods=np.array(list(predictions)), picks=np.stack(list(predictions.values()))
    )
    for picks in predictions.values():
        if not panel.valid[np.arange(len(panel)), picks].all():
            raise RuntimeError("A selector returned an unavailable answer.")
    np.savez_compressed(directory / "predictions.npz", **arrays)
    write_json(cache, {"signature": signature, "n": len(panel), "models": info})
    return arrays


def selections(predictions):
    return dict(zip(predictions["methods"].tolist(), predictions["picks"]))
