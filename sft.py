"""
This module performs supervised finetuning on the OLMo, Gemma, and LLama3 using the
Scientific Abstract-Significance Statement dataset (SASS). It concatenates scientific
abstracts with their simplified versions using a straightforward template.
"""

__author__ = "hw56@indiana.edu"
__version__ = "0.0.1"
__license__ = "0BSD"

import argparse
import os
from typing import List

import torch
import wandb
from datasets import DatasetDict, load_from_disk
from transformers import (AutoModelForCausalLM, AutoModelForSeq2SeqLM,
                          AutoTokenizer, EarlyStoppingCallback,
                          TrainingArguments)
from trl import SFTTrainer, set_seed

OLMO_1B = 'allenai/OLMo-1B-hf'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
DATASET_PATH = 'data/poem_content_h2'  # cleanded from data/poem_content_h2.csv
CKPTS_DIR = 'ckpts'
os.makedirs(CKPTS_DIR, exist_ok=True)
PROJECT_NAME = 'poetry_interpretation'
SEED = 42
TASK_PREFIX = 'Summarize the poem: '
RESPONSE_TEMP = '\nSummary:'

def formatting_func(example: DatasetDict) -> List[str]:
    """
    Formats input examples by concatenating the source text with the target text,
    using the task-specific prefix and response template.

    Args:
        example: A dataset dictionary containing 'source' and 'target' fields.

    Returns:
        A list of formatted strings ready for model training.
    """
    output_texts = []
    for i in range(len(example["source"])):
        text = (
            TASK_PREFIX
            + f"{example['source'][i]}{RESPONSE_TEMP} {example['target'][i]}"
        )
        output_texts.append(text)

    return output_texts


if __name__ == "__main__":

    set_seed(SEED + 2122)
    parser = argparse.ArgumentParser(description="Supervise Fine-tuning with OLMo-1B.")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    args = parser.parse_args()

    # data
    dataset = load_from_disk(DATASET_PATH)

    # model and tokenizer
    model_name = OLMO_1B  # support olmo-1b only
    run_name = f'sft_{model_name.split("/")[-1]}'
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="right")
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 torch_dtype=torch.bfloat16)

    training_args = TrainingArguments(
        output_dir=f"{CKPTS_DIR}/{run_name}",
        overwrite_output_dir=False,
        num_train_epochs=100.0,
        do_train=True,
        do_eval=True,
        do_predict=True,
        evaluation_strategy="steps",
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,  # same to training
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        lr_scheduler_type='constant_with_warmup',
        warmup_steps=50,
        weight_decay=1e-1,
        logging_steps=20,
        eval_steps=20,
        bf16=True,
        report_to="wandb",
        load_best_model_at_end=True,
        save_steps=20,
        save_total_limit=3,
        remove_unused_columns=True,
    )
    wandb.init(project=PROJECT_NAME, name=run_name, config=training_args)

    trainer = SFTTrainer(
        model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        formatting_func=formatting_func,
        max_seq_length=2048,
        args=training_args,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    trainer.train()
