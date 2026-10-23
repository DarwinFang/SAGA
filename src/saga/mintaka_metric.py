"""Alias, numerical, boolean, and date scoring for saved Mintaka references."""

import re
import string
from datetime import datetime
from decimal import Decimal, InvalidOperation
from fractions import Fraction

ARTICLES = re.compile("\\b(a|an|the)\\b", re.I)

NUM = re.compile(
    "^\\s*(?:the answer is\\s+)?(?P<sym>[$£€])?\\s*(?P<num>[+-]?(?:\\d{1,3}(?:,\\d{3})+|\\d+)(?:\\.\\d+)?(?:\\s*/\\s*[+-]?\\d+(?:\\.\\d+)?)?)\\s*(?P<unit>%|percent|percentage|[a-zA-Z]+(?:\\s+[a-zA-Z]+){0,2})?\\s*\\.?\\s*$",
    re.I,
)

LAST_NUM = re.compile(
    "(?<![\\w.])[$£€]?\\s*[+-]?(?:\\d{1,3}(?:,\\d{3})+|\\d+)(?:\\.\\d+)?(?:\\s*/\\s*[+-]?\\d+(?:\\.\\d+)?)?\\s*(?:%|percent|percentage|dollars?|inches?|feet|meters?|cm|hours?|days?|years?)?(?![\\w.])",
    re.I,
)


def norm_text(s):
    s = s.lower()
    s = "".join((ch for ch in s if ch not in string.punctuation))
    s = ARTICLES.sub(" ", s)
    return " ".join(s.split())


def number(s, loose=False):
    m = NUM.match(s.strip())
    if not m and loose:
        matches = list(LAST_NUM.finditer(s))
        if matches:
            m = NUM.match(matches[-1].group(0).strip())
    if not m:
        return None
    raw = m.group("num").replace(",", "").replace(" ", "")
    try:
        if "/" in raw:
            x, y = raw.split("/")
            if Decimal(y) == 0:
                return None
            val = Fraction(Decimal(x)) / Fraction(Decimal(y))
        else:
            val = Fraction(Decimal(raw))
    except (InvalidOperation, ZeroDivisionError, ValueError):
        return None
    unit = (m.group("unit") or "").lower()
    if unit in ("percent", "percentage"):
        unit = "%"
    if m.group("sym"):
        unit = {"$": "dollar", "£": "pound", "€": "euro"}[m.group("sym")]
    return (val, unit)


def unit_compatible(a, b):
    return not b or not a or a == b


def exact(pred, gold, numeric=False):
    pn, gn = (number(pred, loose=numeric), number(gold))
    if numeric:
        return int(
            pn is not None
            and gn is not None
            and (pn[0] == gn[0])
            and unit_compatible(pn[1], gn[1])
        )
    if pn is not None and gn is not None:
        return int(pn[0] == gn[0] and unit_compatible(pn[1], gn[1]))
    return int(norm_text(pred) == norm_text(gold) and bool(norm_text(pred)))


def parse_date(s):
    s = s.strip().rstrip(".").replace("Sept.", "Sep").replace("Sept ", "Sep ")
    s = re.sub("(\\d+)(st|nd|rd|th)\\b", "\\1", s, flags=re.I)
    for pattern, shape in (
        ("%Y-%m-%d", 3),
        ("%Y-%m", 2),
        ("%Y", 1),
        ("%m/%d/%Y", 3),
        ("%m/%d/%y", 3),
        ("%B %d, %Y", 3),
        ("%b %d, %Y", 3),
        ("%B %Y", 2),
        ("%b %Y", 2),
    ):
        try:
            v = datetime.strptime(s, pattern)
            return (
                v.year,
                v.month if shape >= 2 else None,
                v.day if shape >= 3 else None,
            )
        except ValueError:
            pass
    return None


def bool_value(s):
    m = re.match("^\\s*(yes|no|true|false)\\b", s, re.I)
    if not m:
        return None
    return m.group(1).lower() in ("yes", "true")


def aliases(g):
    ans = g["answer"]
    out = [str(ans["mention"])]
    if ans["answerType"] == "entity":
        vals = ans.get("answer")
        if isinstance(vals, list) and len(vals) == 1 and isinstance(vals[0], dict):
            label = vals[0].get("label")
            if isinstance(label, dict) and label.get("en"):
                out.append(label["en"])
    return list(dict.fromkeys(out))


def score_one(pred, g):
    if not pred:
        return 0
    ans = g["answer"]
    typ = ans["answerType"]
    golds = aliases(g)
    if typ == "boolean":
        vals = ans.get("answer")
        target = vals[0] if isinstance(vals, list) and vals else bool_value(golds[0])
        target = target if isinstance(target, bool) else bool_value(str(target))
        correct = int(bool_value(pred) == target and bool_value(pred) is not None)
        return correct
    if typ == "numerical":
        vals = ans.get("answer")
        numeric_gold = str(vals[0]) if isinstance(vals, list) and vals else golds[0]
        target = golds[0] if number(golds[0]) is not None else numeric_gold
        correct = exact(pred, target, True)
        return correct
    if typ == "date":
        vals = ans.get("answer")
        raw = [golds[0]] + ([str(vals[0])] if isinstance(vals, list) and vals else [])
        parsed = parse_date(pred)
        correct = int(parsed is not None and parsed in {parse_date(x) for x in raw})
        return correct
    correct = max((exact(pred, gold, False) for gold in golds))
    return correct
