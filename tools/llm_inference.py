"""Compute UAS likelihoods or four cyclic USC selections on collected answers."""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np

from saga.data import DATASETS, Panel, read_jsonl


def render(tokenizer, prompt, date):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False,
        add_generation_prompt=True, strftime_now=lambda fmt: date.strftime(fmt),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=["uas", "usc", "pack"])
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--config", type=Path, default=Path("configs/paper.json"))
    parser.add_argument("--model-index", type=int, choices=range(4))
    parser.add_argument("--model-path", help="Local checkpoint or Hugging Face model ID")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    folder = args.data / args.dataset
    p = Panel.load(folder)
    instances = {r["id"]: r for r in read_jsonl(folder / "instances.jsonl")}
    dest = folder / "llm_inference"
    dest.mkdir(exist_ok=True)
    if args.method == "pack":
        scores = []
        for k in range(4):
            with np.load(dest / f"likelihood_{k}.npz", allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["ids"], p.ids)
                np.testing.assert_array_equal(saved["sources"], p.sources)
                scores.append(saved["scores"])
        with np.load(dest / "usc.npz", allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["ids"], p.ids)
            choices = saved["choices"]
        np.savez_compressed(folder / "llm_scores.npz", ids=p.ids, sources=p.sources,
                            log_likelihood=np.stack(scores, axis=1), usc_choices=choices)
        return
    date = datetime.fromisoformat(cfg["generation"]["chat_template_date"])
    if args.method == "usc":
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        model_id = args.model_path or cfg["sources"][cfg["supplementary"]["usc_selector"]]["model"]
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        queue = []
        for i, identity in enumerate(p.ids):
            query = instances[str(identity)]["query"]
            if args.dataset == "CrowdWSA":
                query = "Translate this Japanese text into English:\n" + query
            for rotation in range(4):
                order = list(range(rotation, 4)) + list(range(rotation))
                prompt = ("I have generated the following responses to the input:\n" + query + "\n\n"
                          + "\n\n".join(f"Response {j}: {p.texts[i, k]}" for j, k in enumerate(order))
                          + "\n\nEvaluate these responses. Select the most consistent response based on majority consensus.\n"
                          + "Output only the selected response ID: 0, 1, 2, or 3. Do not generate a new answer.")
                text = render(tokenizer, prompt, date)
                if len(tokenizer.encode(text, add_special_tokens=False)) + 16 > 8192:
                    raise ValueError("USC input exceeds the configured context window")
                queue.append((i, rotation, order, text))
        model = LLM(model=model_id, dtype="bfloat16", seed=42, max_model_len=8192,
                    gpu_memory_utilization=.65, max_num_seqs=64, disable_log_stats=True)
        choices = np.full((len(p), 4), -1, dtype=int)
        with (dest / "usc_records.jsonl").open("w") as stream:
            for start in range(0, len(queue), 64):
                batch = queue[start:start + 64]
                outputs = model.generate([r[3] for r in batch], SamplingParams(temperature=0, max_tokens=16, seed=42), use_tqdm=False)
                for (i, rotation, order, _), output in zip(batch, outputs):
                    raw = output.outputs[0].text
                    match = re.fullmatch(r"(?:Response\s*)?([0-3])\s*[.]?", raw.strip(), re.I)
                    if match:
                        choices[i, rotation] = order[int(match[1])]
                    stream.write(json.dumps({"id": str(p.ids[i]), "rotation": rotation,
                                             "order": order, "raw_output": raw,
                                             "selected_source": int(choices[i, rotation])}) + "\n")
        np.savez_compressed(dest / "usc.npz", ids=p.ids, sources=p.sources, choices=choices)
        return
    if args.model_index is None:
        parser.error("uas requires --model-index")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(42)
    model_id = args.model_path or cfg["sources"][args.model_index]["model"]
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, attn_implementation="sdpa", trust_remote_code=True,
    ).to(args.device).eval()
    queue = []
    for i, identity in enumerate(p.ids):
        prompt = render(tokenizer, instances[str(identity)]["prompt"], date)
        pt = tokenizer.encode(prompt, add_special_tokens=False)
        groups = {}
        for k, text in enumerate(p.texts[i]):
            groups.setdefault(str(text), []).append(k)
        for text, slots in groups.items():
            ct = tokenizer.encode(text, add_special_tokens=False)
            sequence = pt + ct
            if not ct or len(sequence) >= model.config.max_position_embeddings:
                raise ValueError("Empty completion or insufficient model context")
            queue.append((len(sequence), i, slots, len(pt), ct, sequence))
    queue.sort(key=lambda row: row[0])
    scores = np.full(p.valid.shape, np.nan)
    with torch.inference_mode():
        for start in range(0, len(queue), 8):
            batch = queue[start:start + 8]
            length = max(row[0] for row in batch)
            ids = torch.full((len(batch), length), tokenizer.pad_token_id, dtype=torch.long, device=args.device)
            mask = torch.zeros_like(ids)
            for j, row in enumerate(batch):
                ids[j, :row[0]] = torch.tensor(row[5], device=args.device)
                mask[j, :row[0]] = 1
            logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
            for j, (_, i, slots, plen, ct, _) in enumerate(batch):
                selected = logits[j, plen - 1:plen - 1 + len(ct)].float()
                value = -torch.nn.functional.cross_entropy(selected, torch.tensor(ct, device=args.device), reduction="mean")
                scores[i, slots] = float(value)
            del logits
    np.savez_compressed(dest / f"likelihood_{args.model_index}.npz", ids=p.ids, sources=p.sources, scores=scores)


if __name__ == "__main__":
    main()
