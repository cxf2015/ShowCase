from __future__ import annotations

import argparse
import math
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler, set_seed

from distill_qwen.config_utils import load_config, load_jsonl, resolve_path
from distill_qwen.student import load_tokenizer, prepare_student_artifacts


@dataclass
class SupervisedCollator:
    tokenizer: Any

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch = self.tokenizer.pad(
            [{"input_ids": item["input_ids"], "attention_mask": item["attention_mask"]} for item in features],
            padding=True,
            return_tensors="pt",
        )

        max_len = batch["input_ids"].shape[1]
        labels = torch.full((len(features), max_len), -100, dtype=torch.long)

        for row_index, item in enumerate(features):
            item_labels = torch.tensor(item["labels"], dtype=torch.long)
            labels[row_index, : item_labels.shape[0]] = item_labels

        batch["labels"] = labels
        return batch


def prepare_example(example: dict[str, Any], tokenizer: Any, max_seq_length: int) -> dict[str, list[int]]:
    prompt_ids = tokenizer(example["prompt"], add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(example["response"], add_special_tokens=False)["input_ids"]
    response_ids = response_ids + [tokenizer.eos_token_id]

    if len(response_ids) >= max_seq_length:
        response_ids = response_ids[: max_seq_length - 1] + [tokenizer.eos_token_id]
        prompt_ids = []
    else:
        max_prompt_tokens = max_seq_length - len(response_ids)
        prompt_ids = prompt_ids[-max_prompt_tokens:]

    input_ids = prompt_ids + response_ids
    labels = ([-100] * len(prompt_ids)) + response_ids
    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def select_precision(training_cfg: dict[str, Any]) -> tuple[bool, bool]:
    if not torch.cuda.is_available():
        return False, False
    bf16 = bool(training_cfg.get("bf16", False)) and torch.cuda.is_bf16_supported()
    fp16 = bool(training_cfg.get("fp16", False)) and not bf16
    return bf16, fp16


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def save_checkpoint(model: Any, tokenizer: Any, output_dir: Path, step: int) -> None:
    checkpoint_dir = output_dir / f"checkpoint-{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir, safe_serialization=True)
    tokenizer.save_pretrained(checkpoint_dir / "tokenizer")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Qwen 0.2B student model.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_cfg = config["dataset"]
    training_cfg = config["training"]
    project_root = config["_project_root"]

    set_seed(int(training_cfg.get("seed", 42)))

    distill_data_path = resolve_path(project_root, dataset_cfg["output_path"])
    if distill_data_path is None or not distill_data_path.exists():
        raise FileNotFoundError(
            "Distillation corpus not found. Generate teacher data first, or point dataset.output_path to an existing JSONL file."
        )
    if len(load_jsonl(distill_data_path)) == 0:
        raise RuntimeError(
            f"Distillation corpus is empty: {distill_data_path}. Teacher generation likely failed before writing any rows."
        )

    student_dir = prepare_student_artifacts(config, tokenizer_corpus_path=distill_data_path)
    tokenizer = load_tokenizer(student_dir)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(student_dir)

    if bool(training_cfg.get("gradient_checkpointing", False)):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    max_seq_length = int(training_cfg.get("max_seq_length", 1024))
    dataset = load_dataset("json", data_files=str(distill_data_path), split="train")
    dataset = dataset.map(
        lambda item: prepare_example(item, tokenizer, max_seq_length),
        remove_columns=dataset.column_names,
        desc="Tokenizing distillation dataset",
    )

    output_dir = resolve_path(project_root, training_cfg["output_dir"])
    if output_dir is None:
        raise ValueError("training.output_dir must be configured.")
    output_dir.mkdir(parents=True, exist_ok=True)

    per_device_batch_size = int(training_cfg.get("per_device_train_batch_size", 1))
    gradient_accumulation_steps = int(training_cfg.get("gradient_accumulation_steps", 1))
    learning_rate = float(training_cfg.get("learning_rate", 2.0e-4))
    num_train_epochs = float(training_cfg.get("num_train_epochs", 1))
    max_steps = int(training_cfg.get("max_steps", 0))
    warmup_ratio = float(training_cfg.get("warmup_ratio", 0.0))
    logging_steps = int(training_cfg.get("logging_steps", 10))
    save_steps = int(training_cfg.get("save_steps", 200))
    weight_decay = float(training_cfg.get("weight_decay", 0.0))
    lr_scheduler_type = str(training_cfg.get("lr_scheduler_type", "cosine"))

    data_loader = DataLoader(
        dataset,
        batch_size=per_device_batch_size,
        shuffle=True,
        collate_fn=SupervisedCollator(tokenizer),
    )

    if len(data_loader) == 0:
        raise RuntimeError("Training dataset is empty after tokenization.")

    updates_per_epoch = math.ceil(len(data_loader) / gradient_accumulation_steps)
    total_training_steps = max_steps if max_steps > 0 else math.ceil(num_train_epochs * updates_per_epoch)
    warmup_steps = int(total_training_steps * warmup_ratio)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = get_scheduler(
        name=lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max(total_training_steps, 1),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    bf16, fp16 = select_precision(training_cfg)
    amp_dtype = torch.bfloat16 if bf16 else torch.float16
    use_amp = device.type == "cuda" and (bf16 or fp16)
    scaler = torch.amp.GradScaler("cuda") if fp16 and device.type == "cuda" else None

    completed_steps = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch_index in range(math.ceil(num_train_epochs)):
        for batch_index, batch in enumerate(data_loader, start=1):
            batch = move_batch_to_device(batch, device)
            autocast_context = (
                torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else nullcontext()
            )

            with autocast_context:
                outputs = model(**batch)
                loss = outputs.loss / gradient_accumulation_steps

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if batch_index % gradient_accumulation_steps != 0:
                continue

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            completed_steps += 1

            if completed_steps % logging_steps == 0 or completed_steps == 1:
                print(
                    f"step={completed_steps} loss={loss.detach().float().item() * gradient_accumulation_steps:.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.6e}"
                )

            if save_steps > 0 and completed_steps % save_steps == 0:
                save_checkpoint(model, tokenizer, output_dir, completed_steps)

            if max_steps > 0 and completed_steps >= max_steps:
                break

        if max_steps > 0 and completed_steps >= max_steps:
            break

    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir, safe_serialization=True)
    tokenizer.save_pretrained(final_dir / "tokenizer")

    print(f"Training finished. Final model saved to: {final_dir}")


if __name__ == "__main__":
    main()