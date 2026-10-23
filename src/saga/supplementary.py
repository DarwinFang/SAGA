"""The supplementary HRRASA, Smoothie, UAS, and USC comparisons."""

from collections import Counter

import numpy as np

from .data import evaluation_texts
from .evaluation import normalize, number, selection_score
from .runner import QUERY, baseline_predictions, selections


def uas_select(texts, log_likelihood, dataset):
    """Author main-entry voting rule with cross-model mean log likelihood."""
    probability = log_likelihood.mean(axis=1)
    if probability.shape != texts.shape or not np.isfinite(probability).all():
        raise ValueError("Expected finite model-by-candidate log likelihoods")
    picks = []
    for answers, scores in zip(texts, probability):
        labels = []
        for k, text in enumerate(answers):
            text = str(text)
            if dataset in ("Mintaka", "PopQA"):
                label = normalize(text) or ("invalid", k)
            elif dataset == "GSM8K":
                value = number(text)
                label = ("number", str(value)) if value is not None else ("unparsed", text)
            else:
                label = text
            labels.append(label)
        counts = Counter(labels)
        mode = max(labels, key=counts.get)
        if counts[mode] == 1:
            mode = labels[int(np.argmax(scores))]
        picks.append(max((k for k, v in enumerate(labels) if v == mode), key=lambda k: scores[k]))
    return np.array(picks)


def supplementary_results(args, load):
    results = {}
    for ds in args.datasets:
        panel, quality, metric, arrays = load(args, ds)
        predictions = baseline_predictions(panel, supplementary=True)
        values = {name: selection_score(quality, predictions[name]) for name in ("HRRASA", "Smoothie")}
        with np.load(args.data / ds / "llm_scores.npz", allow_pickle=False) as saved:
            if not np.array_equal(saved["ids"], panel.ids) or not np.array_equal(saved["sources"], panel.sources):
                raise ValueError("LLM scores are not aligned to candidate IDs and sources")
            uas = uas_select(evaluation_texts(args.data / ds, ds, panel), saved["log_likelihood"], ds)
            usc = saved["usc_choices"]
        if usc.shape != (len(panel), 4) or not np.isin(usc, [-1, 0, 1, 2, 3]).all():
            raise ValueError("Expected four cyclic USC selections per input")
        values["UAS"] = selection_score(quality, uas)
        rotations = []
        for column in usc.T:
            valid = column >= 0
            selected = np.zeros(len(panel))
            selected[valid] = quality[np.flatnonzero(valid), column[valid]]
            rotations.append(float(selected.mean()))
        values["USC"] = float(np.mean(rotations))
        values["SAGA"] = selection_score(quality, selections(arrays)[QUERY])
        results[ds] = {"metric": metric, "scores": values, "usc_order_scores": rotations}
    return results
