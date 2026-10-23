"""Generate a source's responses with the paper task prompts and token limits."""

import argparse
import json
from datetime import datetime
from pathlib import Path

from saga.data import DATASETS, read_jsonl
from saga.evaluation import extract_final_answer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--source", type=int, required=True, choices=range(4))
    parser.add_argument("--model-path")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--config", type=Path, default=Path("configs/paper.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    cfg = json.loads(args.config.read_text())
    generation = cfg["generation"]
    spec = cfg["sources"][args.source]
    model_id = args.model_path or spec["model"]
    date = datetime.fromisoformat(generation["chat_template_date"])
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    records = read_jsonl(args.data / args.dataset / "instances.jsonl")
    prompts = [tokenizer.apply_chat_template(
        [{"role": "user", "content": r["prompt"]}], tokenize=False,
        add_generation_prompt=True, strftime_now=lambda fmt: date.strftime(fmt),
    ) for r in records]
    limit = generation["max_tokens"][args.dataset]
    model = LLM(model=model_id, dtype="bfloat16", seed=generation["seed"], trust_remote_code=True)
    outputs = model.generate(prompts, SamplingParams(temperature=generation["temperature"],
                                                     max_tokens=limit, seed=generation["seed"]))
    limits = [limit] * len(records)
    if args.dataset == "GSM8K":
        retry = [i for i, output in enumerate(outputs)
                 if output.outputs[0].finish_reason == "length"
                 and extract_final_answer(output.outputs[0].text)[1] == "no_final_prefix_retained"]
        if retry:
            limit_retry = generation["gsm8k_retry_tokens"]
            repeated = model.generate([prompts[i] for i in retry], SamplingParams(
                temperature=generation["temperature"], max_tokens=limit_retry, seed=generation["seed"]))
            for i, output in zip(retry, repeated):
                outputs[i] = output
                limits[i] = limit_retry
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        for instance, output, used_limit in zip(records, outputs, limits):
            generated = output.outputs[0]
            answer, status = extract_final_answer(generated.text) if args.dataset == "GSM8K" else (generated.text.strip(), "text")
            record = {"id": instance["id"], "source": spec["name"], "slot": args.source,
                      "raw_output": generated.text, "answer": answer, "parse_status": status,
                      "stop_reason": generated.finish_reason, "max_tokens": used_limit,
                      "temperature": generation["temperature"], "seed": generation["seed"],
                      "truncated_without_final": args.dataset == "GSM8K" and generated.finish_reason == "length" and status == "no_final_prefix_retained"}
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
