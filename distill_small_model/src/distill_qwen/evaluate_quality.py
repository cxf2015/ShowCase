from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from distill_qwen.config_utils import ensure_parent_dir, load_config, load_jsonl, resolve_path
from distill_qwen.student import load_tokenizer, prepare_student_artifacts
from openai import OpenAI


TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+|[^\s]")


@dataclass
class GenerationBundle:
    model: Any
    tokenizer: Any
    device: torch.device


def normalize_tokens(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def lcs_length(first: list[str], second: list[str]) -> int:
    if not first or not second:
        return 0

    previous = [0] * (len(second) + 1)
    for token_a in first:
        current = [0]
        for index_b, token_b in enumerate(second, start=1):
            if token_a == token_b:
                current.append(previous[index_b - 1] + 1)
            else:
                current.append(max(current[-1], previous[index_b]))
        previous = current
    return previous[-1]


def rouge_l_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_tokens(prediction)
    ref_tokens = normalize_tokens(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_tokens(prediction)
    ref_tokens = normalize_tokens(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0

    ref_counts: dict[str, int] = {}
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1

    overlap = 0
    for token in pred_tokens:
        count = ref_counts.get(token, 0)
        if count > 0:
            overlap += 1
            ref_counts[token] = count - 1

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, reference: str) -> float:
    normalized_prediction = " ".join(normalize_tokens(prediction))
    normalized_reference = " ".join(normalize_tokens(reference))
    return 1.0 if normalized_prediction == normalized_reference else 0.0


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def load_eval_records(eval_path: Path, max_eval_samples: int) -> list[dict[str, Any]]:
    records = load_jsonl(eval_path)
    if max_eval_samples > 0:
        records = records[:max_eval_samples]
    if not records:
        raise RuntimeError("Evaluation dataset is empty.")
    for record in records:
        if "prompt" not in record:
            raise ValueError("Every evaluation record must contain a 'prompt' field.")
    return records


def resolve_baseline_path(config: dict[str, Any], evaluation_cfg: dict[str, Any], eval_path: Path) -> Path:
    project_root = config["_project_root"]
    requested = resolve_path(project_root, evaluation_cfg.get("baseline_model_path"))
    if requested and requested.exists():
        return requested
    return prepare_student_artifacts(config, tokenizer_corpus_path=eval_path)


def resolve_distilled_path(config: dict[str, Any], evaluation_cfg: dict[str, Any]) -> Path:
    project_root = config["_project_root"]
    requested = resolve_path(project_root, evaluation_cfg.get("distilled_model_path"))
    if requested is None or not requested.exists():
        raise FileNotFoundError("Distilled model path does not exist. Train the student first or adjust evaluation.distilled_model_path.")
    return requested


def load_generation_bundle(model_path: Path) -> GenerationBundle:
    tokenizer_path = model_path / "tokenizer"
    tokenizer_source = tokenizer_path if tokenizer_path.exists() else model_path
    tokenizer = load_tokenizer(tokenizer_source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return GenerationBundle(model=model, tokenizer=tokenizer, device=device)


def generate_text(bundle: GenerationBundle, prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> str:
    encoded = bundle.tokenizer(prompt, return_tensors="pt")
    encoded = {
        key: value.to(bundle.device)
        for key, value in encoded.items()
        if key in {"input_ids", "attention_mask"}
    }

    do_sample = temperature > 0
    with torch.no_grad():
        generated = bundle.model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            pad_token_id=bundle.tokenizer.pad_token_id,
            eos_token_id=bundle.tokenizer.eos_token_id,
        )

    new_tokens = generated[0][encoded["input_ids"].shape[1] :]
    return bundle.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def build_judge_client(config: dict[str, Any]) -> OpenAI:
    teacher_cfg = config["teacher"]
    api_key_env = teacher_cfg["api_key_env"]
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(f"Environment variable '{api_key_env}' is not set.")
    return OpenAI(api_key=api_key, base_url=teacher_cfg["api_base"])


def judge_pairwise(
    client: OpenAI,
    config: dict[str, Any],
    prompt: str,
    baseline_answer: str,
    distilled_answer: str,
    reference: str | None,
    max_new_tokens: int,
) -> dict[str, str]:
    teacher_cfg = config["teacher"]
    lines = [
        "You are an impartial evaluator comparing two model answers.",
        "Return exactly two lines.",
        "Line 1 format: WINNER: baseline|distilled|tie",
        "Line 2 format: REASON: one short sentence",
        "Prefer factual accuracy, instruction following, completeness, and clarity.",
        "",
        f"PROMPT:\n{prompt}",
    ]
    if reference:
        lines.extend(["", f"REFERENCE ANSWER:\n{reference}"])
    lines.extend(
        [
            "",
            f"ANSWER A (baseline):\n{baseline_answer}",
            "",
            f"ANSWER B (distilled):\n{distilled_answer}",
        ]
    )
    judge_prompt = "\n".join(lines)

    response = client.chat.completions.create(
        model=teacher_cfg["model_name"],
        messages=[
            {"role": "system", "content": "You are a strict evaluation judge."},
            {"role": "user", "content": judge_prompt},
        ],
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_new_tokens,
    )
    content = (response.choices[0].message.content or "").strip()
    winner = "tie"
    reason = content
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("WINNER:"):
            winner = stripped.split(":", 1)[1].strip().lower()
        if stripped.upper().startswith("REASON:"):
            reason = stripped.split(":", 1)[1].strip()
    if winner not in {"baseline", "distilled", "tie"}:
        winner = "tie"
    return {"winner": winner, "reason": reason}


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    baseline_rouge = [item["baseline_metrics"]["rouge_l_f1"] for item in records if item.get("baseline_metrics")]
    distilled_rouge = [item["distilled_metrics"]["rouge_l_f1"] for item in records if item.get("distilled_metrics")]
    baseline_f1 = [item["baseline_metrics"]["token_f1"] for item in records if item.get("baseline_metrics")]
    distilled_f1 = [item["distilled_metrics"]["token_f1"] for item in records if item.get("distilled_metrics")]
    baseline_em = [item["baseline_metrics"]["exact_match"] for item in records if item.get("baseline_metrics")]
    distilled_em = [item["distilled_metrics"]["exact_match"] for item in records if item.get("distilled_metrics")]

    judge_results = [item["judge"] for item in records if item.get("judge")]
    baseline_wins = sum(1 for item in judge_results if item["winner"] == "baseline")
    distilled_wins = sum(1 for item in judge_results if item["winner"] == "distilled")
    ties = sum(1 for item in judge_results if item["winner"] == "tie")

    return {
        "sample_count": len(records),
        "baseline": {
            "avg_rouge_l_f1": average(baseline_rouge),
            "avg_token_f1": average(baseline_f1),
            "avg_exact_match": average(baseline_em),
        },
        "distilled": {
            "avg_rouge_l_f1": average(distilled_rouge),
            "avg_token_f1": average(distilled_f1),
            "avg_exact_match": average(distilled_em),
        },
        "judge": {
            "baseline_wins": baseline_wins,
            "distilled_wins": distilled_wins,
            "ties": ties,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate answer quality before and after distillation.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    args = parser.parse_args()

    config = load_config(args.config)
    project_root = config["_project_root"]
    training_cfg = config.get("training", {})
    evaluation_cfg = config["evaluation"]
    set_seed(int(training_cfg.get("seed", 42)))

    eval_path = resolve_path(project_root, evaluation_cfg["eval_data_path"])
    if eval_path is None or not eval_path.exists():
        raise FileNotFoundError("Evaluation dataset path does not exist.")

    output_dir = resolve_path(project_root, evaluation_cfg["output_dir"])
    if output_dir is None:
        raise ValueError("evaluation.output_dir must be configured.")
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_eval_records(eval_path, int(evaluation_cfg.get("max_eval_samples", 0)))
    baseline_path = resolve_baseline_path(config, evaluation_cfg, eval_path)
    distilled_path = resolve_distilled_path(config, evaluation_cfg)

    print(f"Loading baseline model from: {baseline_path}")
    baseline_bundle = load_generation_bundle(baseline_path)
    print(f"Loading distilled model from: {distilled_path}")
    distilled_bundle = load_generation_bundle(distilled_path)

    judge_client = None
    if bool(evaluation_cfg.get("judge_with_teacher", False)):
        judge_client = build_judge_client(config)

    max_new_tokens = int(evaluation_cfg.get("max_new_tokens", 256))
    temperature = float(evaluation_cfg.get("temperature", 0.0))
    top_p = float(evaluation_cfg.get("top_p", 1.0))
    judge_max_new_tokens = int(evaluation_cfg.get("judge_max_new_tokens", 256))

    generations_path = output_dir / "generations.jsonl"
    summary_path = output_dir / "summary.json"
    ensure_parent_dir(generations_path)

    written_records: list[dict[str, Any]] = []
    with generations_path.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(records):
            prompt = record["prompt"]
            reference = record.get("reference") or record.get("response")

            baseline_answer = generate_text(baseline_bundle, prompt, max_new_tokens, temperature, top_p)
            distilled_answer = generate_text(distilled_bundle, prompt, max_new_tokens, temperature, top_p)

            item: dict[str, Any] = {
                "index": index,
                "prompt": prompt,
                "reference": reference,
                "baseline_answer": baseline_answer,
                "distilled_answer": distilled_answer,
            }

            if reference:
                item["baseline_metrics"] = {
                    "rouge_l_f1": rouge_l_f1(baseline_answer, reference),
                    "token_f1": token_f1(baseline_answer, reference),
                    "exact_match": exact_match(baseline_answer, reference),
                    "answer_length": len(normalize_tokens(baseline_answer)),
                }
                item["distilled_metrics"] = {
                    "rouge_l_f1": rouge_l_f1(distilled_answer, reference),
                    "token_f1": token_f1(distilled_answer, reference),
                    "exact_match": exact_match(distilled_answer, reference),
                    "answer_length": len(normalize_tokens(distilled_answer)),
                }

            if judge_client is not None:
                item["judge"] = judge_pairwise(
                    client=judge_client,
                    config=config,
                    prompt=prompt,
                    baseline_answer=baseline_answer,
                    distilled_answer=distilled_answer,
                    reference=reference,
                    max_new_tokens=judge_max_new_tokens,
                )

            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            written_records.append(item)
            print(f"evaluated sample {index + 1}/{len(records)}")

    summary = summarize_records(written_records)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved detailed generations to: {generations_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()