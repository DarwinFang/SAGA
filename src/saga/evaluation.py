"""Dataset metrics, applied after aggregation."""

import re
import string
from fractions import Fraction

import numpy as np

from .data import evaluation_texts, read_jsonl


def normalize(text):
    text = "".join(c for c in text.lower() if c not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def number(text):
    value = (
        text.strip()
        .replace("−", "-")
        .replace("\\,", "")
        .replace("\\!", "")
        .replace("\\$", "$")
    )
    value = (
        re.sub(r"\\(?:text|mathrm)\{[^{}]*\}", "", value)
        .strip()
        .strip("$ ")
        .replace(",", "")
        .strip()
    )
    value = re.sub(r"^\\(?:dfrac|tfrac)", r"\\frac", value)
    match = re.fullmatch(
        r"([+-]?)\\frac\{([+-]?\d+(?:\.\d+)?)\}\{([+-]?\d+(?:\.\d+)?)\}", value
    )
    try:
        if match:
            return (
                (-1 if match[1] == "-" else 1) * Fraction(match[2]) / Fraction(match[3])
            )
        if not re.fullmatch(
            r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:\s*/\s*[+-]?(?:\d+(?:\.\d*)?|\.\d+))?",
            value,
        ):
            return None
        if "/" in value:
            left, right = value.split("/")
            return Fraction(left.strip()) / Fraction(right.strip())
        return Fraction(value)
    except (ValueError, ZeroDivisionError):
        return None


def extract_final_answer(raw):
    text, boxes = raw.strip(), []
    for match in re.finditer(r"\\boxed\s*\{", text):
        start = end = match.end()
        depth = 1
        while end < len(text) and depth:
            depth += (text[end] == "{") - (text[end] == "}")
            end += 1
        if depth == 0:
            boxes.append(text[start : end - 1].strip())
    if boxes:
        return boxes[-1], "boxed"
    finals = re.findall(r"(?:FINAL:|Final answer:|####)\s*(.*)", text, re.I)
    if finals:
        return finals[-1].strip(), "final_line"
    return (text.splitlines()[-1] if text else ""), "no_final_prefix_retained"


def score_candidates(directory, dataset, panel):
    texts = evaluation_texts(directory, dataset, panel)
    references = {r["id"]: r for r in read_jsonl(directory / "references.jsonl")}
    values = np.full(panel.valid.shape, np.nan)
    if dataset == "CrowdWSA":
        from nltk.tokenize import TreebankWordTokenizer
        from nltk.translate.gleu_score import sentence_gleu

        tokenize = TreebankWordTokenizer().tokenize
    truncated = set()
    if dataset == "GSM8K":
        truncated = {
            (r["id"], r["slot"])
            for r in read_jsonl(directory / "candidates.jsonl")
            if r.get("truncated_without_final", False)
        }
    for i, identity in enumerate(panel.ids):
        reference = references[str(identity)]
        for k in np.flatnonzero(panel.valid[i]):
            text = str(texts[i, k])
            if dataset == "Mintaka":
                from .mintaka_metric import score_one

                value = 100 * score_one(text, reference)
            elif dataset == "PopQA":
                value = 100 * max(
                    normalize(text) == normalize(a) for a in reference["answers"]
                )
            elif dataset == "CrowdWSA":
                value = 100 * sentence_gleu(
                    [tokenize(reference["english"])], tokenize(text)
                )
            elif dataset == "GSM8K":
                predicted, gold = number(text), number(reference["answer"])
                if gold is None:
                    raise ValueError("Invalid GSM8K reference number")
                value = 100 * float(
                    (str(identity), int(k)) not in truncated
                    and predicted is not None
                    and predicted == gold
                )
            else:
                raise ValueError(dataset)
            values[i, k] = value
    return values


def selection_score(quality, picks):
    return float(quality[np.arange(len(picks)), picks].mean())
