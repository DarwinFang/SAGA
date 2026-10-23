"""Encode the supplied paper inputs and candidates with frozen E5-large-v2."""

import argparse
import json
from pathlib import Path

import numpy as np

from saga.data import DATASETS, Panel, read_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--config", type=Path, default=Path("configs/paper.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    import torch
    from transformers import AutoModel, AutoTokenizer

    config = json.loads(args.config.read_text())["encoder"]
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(
        config["name"], revision=config["revision"]
    )
    model = (
        AutoModel.from_pretrained(
            config["name"], revision=config["revision"], torch_dtype=torch.float32
        )
        .to(args.device)
        .eval()
    )
    panel = Panel.load(args.data / args.dataset)
    instances = {
        r["id"]: r for r in read_jsonl(args.data / args.dataset / "instances.jsonl")
    }
    queries = [config["prefix"] + instances[str(i)]["query"].strip() for i in panel.ids]
    texts = sorted(
        set(queries)
        | {config["prefix"] + str(t).strip() for t in panel.texts[panel.valid]}
    )
    vectors = {}
    with torch.inference_mode():
        batch_size = 16 if args.dataset == "GSM8K" else 32
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=config["max_length"],
                return_tensors="pt",
            ).to(args.device)
            hidden = model(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            embedded = torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().numpy()
            vectors.update(zip(batch, embedded))
    answers = np.zeros((*panel.valid.shape, model.config.hidden_size), dtype=np.float32)
    for i, k in np.argwhere(panel.valid):
        answers[i, k] = vectors[config["prefix"] + str(panel.texts[i, k]).strip()]
    arrays = {name: getattr(panel, name) for name in panel.__dataclass_fields__}
    arrays.update(queries=np.stack([vectors[t] for t in queries]), answers=answers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError("Choose a new output file")
    np.savez_compressed(args.output, **arrays)


if __name__ == "__main__":
    main()
