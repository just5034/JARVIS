"""Physics brain SFT training script — QDoRA on R1-Distill-Qwen-32B.

Phase 4B — fine-tune the base physics model on filtered teacher traces
using QDoRA (Quantized DoRA, rank-32). Requires DeepSpeed ZeRO-3 on
4× A100-40GB.

This script is called by scripts/run_physics_sft.sh via DeepSpeed launcher.
It can also be used for code brain SFT with different data and base model.

Usage (via deepspeed):
    deepspeed --num_gpus=4 -m training.physics.run_sft \
        --model_name_or_path /projects/.../r1-distill-qwen-32b \
        --train_data /scratch/.../physics_filtered_100k.jsonl \
        --output_dir /scratch/.../checkpoints/physics_sft/ \
        --deepspeed configs/ds_zero3.json \
        --use_dora true --lora_rank 32 --lora_alpha 64
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset


@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "Path to base model"})
    use_dora: bool = field(default=True, metadata={"help": "Use DoRA instead of LoRA"})
    lora_rank: int = field(default=32, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=64, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout"})
    lora_target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "Comma-separated target modules for LoRA"},
    )


@dataclass
class DataArguments:
    train_data: str = field(metadata={"help": "Path to training JSONL"})
    max_seq_length: int = field(default=8192, metadata={"help": "Max sequence length"})
    val_split: float = field(default=0.02, metadata={"help": "Fraction for validation"})


@dataclass
class AimArguments:
    aim_repo: str = field(default="/scratch/bgde-delta-gpu/aim", metadata={"help": "Aim repo"})
    aim_experiment: str = field(default="sft", metadata={"help": "Aim experiment name"})


class TraceDataset(Dataset):
    """Dataset of teacher reasoning traces for SFT."""

    def __init__(self, data_path: str, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        with open(data_path) as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    self.examples.append(entry)

        print(f"[dataset] loaded {len(self.examples)} training examples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        entry = self.examples[idx]

        # Format: problem → trace (the model learns to produce the reasoning)
        problem = entry.get("problem", "")
        trace = entry.get("trace", "")
        reasoning = entry.get("reasoning", "")

        # If there's a separate reasoning field (R1 format), prepend it
        if reasoning:
            full_response = f"<think>\n{reasoning}\n</think>\n\n{trace}"
        else:
            full_response = trace

        # Build chat format
        messages = [
            {"role": "user", "content": problem},
            {"role": "assistant", "content": full_response},
        ]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].squeeze()
        attention_mask = encoded["attention_mask"].squeeze()

        # Labels = input_ids (causal LM), mask the prompt portion
        labels = input_ids.clone()

        # Find where the assistant response starts and mask everything before
        prompt_text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": problem}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_len = len(self.tokenizer(prompt_text, truncation=True, max_length=self.max_length)["input_ids"])
        labels[:prompt_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, AimArguments))
    model_args, data_args, training_args, aim_args = parser.parse_args_into_dataclasses()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype="auto",
    )

    # Apply LoRA/DoRA
    target_modules = [m.strip() for m in model_args.lora_target_modules.split(",")]
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=model_args.lora_rank,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        target_modules=target_modules,
        use_dora=model_args.use_dora,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset
    full_dataset = TraceDataset(data_args.train_data, tokenizer, data_args.max_seq_length)

    # Train/val split
    val_size = int(len(full_dataset) * data_args.val_split)
    train_size = len(full_dataset) - val_size

    import torch
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    # Aim callback
    callbacks = []
    try:
        from training.utils.tracking import get_aim_callback
        aim_cb = get_aim_callback(
            experiment=aim_args.aim_experiment,
            aim_repo=aim_args.aim_repo,
        )
        if aim_cb:
            callbacks.append(aim_cb)
    except ImportError:
        print("[sft] aim not available — metrics logged to tensorboard only")

    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=callbacks,
    )

    trainer.train()

    # Save final adapter
    final_dir = Path(training_args.output_dir) / "final"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"\n[sft] adapter saved to {final_dir}")


if __name__ == "__main__":
    main()
