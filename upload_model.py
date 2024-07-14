"""
This script uploads a model checkpoint to the Hugging Face Hub.
"""

__author__ = "hw56@indiana.edu"
__version__ = "0.0.1"
__license__ = "0BSD"

import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, HfApi

def upload_checkpoint(checkpoint_path: str, model_name: str):
    """
    Uploads a checkpoint to the Hugging Face Hub.

    Args:
        checkpoint_path (str): Path to the model checkpoint directory.
        model_name (str): Name of the model.
    """
    # get user information
    api = HfApi()
    user_info = api.whoami()
    username = user_info['name']

    # construct repository URL
    repo_url = f"https://huggingface.co/{username}/{model_name}"

    # load model in bf16 and tokenizer
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path,
                                                 torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    # push model and tokenizer to the repository
    model.push_to_hub(model_name)
    tokenizer.push_to_hub(model_name)

    print(f"Model and tokenizer uploaded to {repo_url}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload checkpoint to Hugging Face Hub.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the model checkpoint directory")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model")

    args = parser.parse_args()

    upload_checkpoint(args.checkpoint_path, args.model_name)
