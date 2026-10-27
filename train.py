"""LoRA supervised fine-tuning (next-token prediction) of Qwen2.5-VL."""

from dataclasses import dataclass, field

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, HfArgumentParser, Qwen2_5_VLForConditionalGeneration, Trainer, TrainingArguments

from data import SFTCollator, SFTDataset

# Linear layers of the language model; the vision tower ("visual.*") is excluded.
LLM_LINEAR = r"^(?!.*visual).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"


@dataclass
class Args:
    model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    data_path: str = field(default=None, metadata={"help": "Training JSON (see data.SFTDataset)."})
    image_folder: str = ""
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    train_projector: bool = True


def main():
    args, training_args = HfArgumentParser((Args, TrainingArguments)).parse_args_into_dataclasses()
    training_args.remove_unused_columns = False

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float32,
        attn_implementation="sdpa",
    )
    processor = AutoProcessor.from_pretrained(args.model_path)
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    # PEFT freezes everything except the LoRA weights and modules_to_save (the vision-to-LLM projector).
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=LLM_LINEAR,
        modules_to_save=["merger"] if args.train_projector else None,
        task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=SFTDataset(args.data_path, args.image_folder),
        data_collator=SFTCollator(processor),
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()
