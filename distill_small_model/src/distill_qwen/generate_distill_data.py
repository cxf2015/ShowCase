from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset
from tqdm import tqdm

from distill_qwen.config_utils import ensure_parent_dir, load_config, load_jsonl, resolve_path
from distill_qwen.prompting import format_prompt
from distill_qwen.teacher_api import MultiTeacherRouter, OpenAICompatibleTeacher


def build_teacher_router(teacher_cfg: dict[str, Any]) -> MultiTeacherRouter:
    model_names = [teacher_cfg["model_name"], *list(teacher_cfg.get("fallback_model_names") or [])]
    teachers = [
        OpenAICompatibleTeacher(
            api_base=teacher_cfg["api_base"],
            api_key_env=teacher_cfg["api_key_env"],
            model_name=model_name,
            system_prompt=teacher_cfg.get("system_prompt"),
            temperature=float(teacher_cfg.get("temperature", 0.3)),
            top_p=float(teacher_cfg.get("top_p", 0.95)),
            max_new_tokens=int(teacher_cfg.get("max_new_tokens", 512)),
            max_retries=int(teacher_cfg.get("max_retries", 6)),
            retry_backoff_seconds=float(teacher_cfg.get("retry_backoff_seconds", 5.0)),
            retry_max_backoff_seconds=float(teacher_cfg.get("retry_max_backoff_seconds", 60.0)),
        )
        for model_name in model_names
    ]
    return MultiTeacherRouter(teachers)


def load_prompt_records(dataset_cfg: dict[str, Any], project_root: str) -> list[dict[str, Any]]:
    local_prompts_path = resolve_path(project_root, dataset_cfg.get("local_prompts_path"))
    if local_prompts_path:
        return load_jsonl(local_prompts_path)

    dataset_name = dataset_cfg.get("hf_dataset")
    if not dataset_name:
        raise ValueError("Either dataset.local_prompts_path or dataset.hf_dataset must be configured.")

    dataset = load_dataset(
        dataset_name,
        dataset_cfg.get("hf_subset"),
        split=dataset_cfg.get("split", "train"),
    )
    take_samples = int(dataset_cfg.get("take_samples") or 0)
    if take_samples > 0:
        dataset = dataset.select(range(min(take_samples, len(dataset))))
    return list(dataset)


def count_existing_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def iter_pending_records(records: list[dict[str, Any]], completed: int) -> Iterable[tuple[int, dict[str, Any]]]:
    for index, record in enumerate(records):
        if index < completed:
            continue
        yield index, record


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate teacher-labeled distillation data.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    args = parser.parse_args()

    config = load_config(args.config)
    teacher_cfg = config["teacher"]
    dataset_cfg = config["dataset"]
    project_root = config["_project_root"]

    output_path = resolve_path(project_root, dataset_cfg["output_path"])
    if output_path is None:
        raise ValueError("dataset.output_path must be configured.")

    ensure_parent_dir(output_path)

    teacher_router = build_teacher_router(teacher_cfg)

    records = load_prompt_records(dataset_cfg, project_root)
    request_interval = float(dataset_cfg.get("request_interval_seconds", 0.0))
    template = dataset_cfg.get("prompt_template", "alpaca")
    completed = count_existing_lines(output_path)

    if completed >= len(records):
        print(f"Teacher data already exists and is complete: {output_path}")
        return

    with output_path.open("a", encoding="utf-8") as handle:
        for index, record in tqdm(iter_pending_records(records, completed), total=max(len(records) - completed, 0)):
            prompt = format_prompt(record, template=template)
            try:
                response, teacher_model_name = teacher_router.generate(prompt)
            except Exception as error:
                raise RuntimeError(
                    f"Teacher generation failed at sample {index}. Partial data has been kept in '{output_path}'. "
                    f"You can retry the same command to resume from the next unfinished row."
                ) from error
            payload = {
                "index": index,
                "prompt": prompt,
                "response": response,
                "teacher_model": teacher_model_name,
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            handle.flush()
            if request_interval > 0:
                time.sleep(request_interval)

    print(f"Saved teacher responses to: {output_path}")


if __name__ == "__main__":
    main()
