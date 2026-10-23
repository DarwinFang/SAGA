"""Evaluation bundles with references kept separate from model inputs."""

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

DATASETS = ("Mintaka", "PopQA", "CrowdWSA", "GSM8K")


def read_jsonl(path):
    return [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass
class Panel:
    ids: np.ndarray
    sources: np.ndarray
    queries: np.ndarray
    answers: np.ndarray
    texts: np.ndarray
    valid: np.ndarray
    source_indices: np.ndarray
    groups: np.ndarray

    @classmethod
    def load(cls, directory):
        path = Path(directory) / "inputs.npz"
        with np.load(path, allow_pickle=False) as data:
            panel = cls(**{name: data[name] for name in cls.__dataclass_fields__})
        n, width = panel.valid.shape
        if any(
            len(getattr(panel, key)) != n
            for key in cls.__dataclass_fields__
            if key != "sources"
        ):
            raise ValueError("Input arrays have different instance counts.")
        if panel.answers.shape[:2] != (n, width) or not panel.valid.any(1).all():
            raise ValueError("Invalid candidate panel.")
        if not np.isfinite(panel.answers).all() or not np.isfinite(panel.queries).all():
            raise ValueError("Representations must be finite.")
        if not np.allclose(
            np.linalg.norm(panel.answers[panel.valid], axis=1), 1, atol=1e-5
        ):
            raise ValueError("Available answer representations must be unit vectors.")
        if (panel.source_indices < 0).any() or (
            panel.source_indices >= len(panel.sources)
        ).any():
            raise ValueError("Candidate source index is out of range.")
        return panel

    def subset(self, indices):
        return replace(
            self,
            **{
                name: getattr(self, name)[indices]
                for name in self.__dataclass_fields__
                if name != "sources"
            },
        )

    @property
    def gram(self):
        answers = self.answers.astype(float)
        return answers @ answers.swapaxes(1, 2)

    def __len__(self):
        return len(self.ids)


def load_quality(directory, panel):
    with np.load(Path(directory) / "scores.npz", allow_pickle=False) as data:
        if not np.array_equal(data["ids"], panel.ids):
            raise ValueError("Candidate scores are not aligned to input IDs.")
        quality = data["quality"]
        metric = str(data["metric"])
    if (
        quality.shape != panel.valid.shape
        or not np.isfinite(quality[panel.valid]).all()
    ):
        raise ValueError("Missing candidate scores.")
    return quality, metric


def evaluation_texts(directory, dataset, panel):
    """GSM8K evaluates extracted answers separately from solution embeddings."""
    if dataset != "GSM8K":
        return panel.texts
    with np.load(Path(directory) / "final_answers.npz", allow_pickle=False) as data:
        if not np.array_equal(data["ids"], panel.ids) or not np.array_equal(data["sources"], panel.sources):
            raise ValueError("Final answers do not match the candidate panel")
        texts = data["texts"]
    if texts.shape != panel.texts.shape:
        raise ValueError("Final answer shape does not match solutions")
    return texts
