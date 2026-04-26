from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
from transformers import AutoTokenizer, PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

from distill_qwen.config_utils import load_jsonl, resolve_path


def load_tokenizer(tokenizer_name_or_path: str | Path):
    try:
        return AutoTokenizer.from_pretrained(tokenizer_name_or_path, fix_mistral_regex=True)
    except TypeError:
        return AutoTokenizer.from_pretrained(tokenizer_name_or_path)


def build_local_tokenizer(corpus_path: Path, vocab_size: int, max_length: int) -> PreTrainedTokenizerFast:
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=1,
        special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
    )

    rows = load_jsonl(corpus_path)

    def text_iterator() -> list[str]:
        texts: list[str] = []
        for item in rows:
            prompt = (item.get("prompt") or "").strip()
            response = (item.get("response") or "").strip()
            if prompt:
                texts.append(prompt)
            if response:
                texts.append(response)
        return texts

    tokenizer.train_from_iterator(text_iterator(), trainer=trainer)

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<unk>",
    )
    fast_tokenizer.model_max_length = max_length
    return fast_tokenizer


def prepare_student_artifacts(config: dict, tokenizer_corpus_path: Path | None = None) -> Path:
    project_root = config["_project_root"]
    student_cfg = config["student"]
    output_dir = resolve_path(project_root, student_cfg["model_output_dir"])
    if output_dir is None:
        raise ValueError("student.model_output_dir must be configured.")

    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.json"

    tokenizer_name = student_cfg.get("tokenizer_name")
    if tokenizer_name:
        tokenizer = load_tokenizer(tokenizer_name)
    else:
        if tokenizer_corpus_path is None:
            raise ValueError("A tokenizer corpus path is required when student.tokenizer_name is null.")
        tokenizer = build_local_tokenizer(
            corpus_path=tokenizer_corpus_path,
            vocab_size=int(student_cfg.get("local_tokenizer_vocab_size", 4096)),
            max_length=int(student_cfg["max_position_embeddings"]),
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = int(student_cfg["max_position_embeddings"])

    if not config_path.exists():
        model_config = Qwen2Config(
            vocab_size=len(tokenizer),
            hidden_size=int(student_cfg["hidden_size"]),
            intermediate_size=int(student_cfg["intermediate_size"]),
            num_hidden_layers=int(student_cfg["num_hidden_layers"]),
            num_attention_heads=int(student_cfg["num_attention_heads"]),
            num_key_value_heads=int(student_cfg["num_key_value_heads"]),
            max_position_embeddings=int(student_cfg["max_position_embeddings"]),
            rope_theta=float(student_cfg.get("rope_theta", 1000000.0)),
            rms_norm_eps=float(student_cfg.get("rms_norm_eps", 1.0e-6)),
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            tie_word_embeddings=False,
        )
        model = Qwen2ForCausalLM(model_config)
        model.save_pretrained(output_dir, safe_serialization=True)
        tokenizer.save_pretrained(output_dir)
    else:
        tokenizer.save_pretrained(output_dir)

    return output_dir
