"""Script to prepare DeepScaler training and test datasets.

This script processes math problem datasets into a standardized format for training
and testing DeepScaler models. It loads problems from specified datasets, adds
instruction prompts, and saves the processed data as parquet files.
"""

import argparse
import os
from typing import Dict, List, Optional, Any

import pandas as pd
from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed

import json
import random

def load_dataset(local_dir: str) -> List[Dict[str, Any]]:
    with open(local_dir, 'r') as f:
        data = json.load(f)
    return data


def make_map_fn(split: str):
    """Create a mapping function to process dataset examples.

    Args:
        split: Dataset split name ('train' or 'test')

    Returns:
        Function that processes individual dataset examples
    """
    def process_fn(example: Dict[str, Any], idx: int, data_source: str) -> Optional[Dict[str, Any]]:
        question = example.pop('problem')
        instruction = "Let's think step by step and output the final answer within \\boxed{}."
        question = f"{question} {instruction}"
        answer = example.pop('answer')

        data = {
            "data_source": data_source,
            "prompt": [{
                "role": "user",
                "content": question
            }],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": answer
            },
            "extra_info": {
                'split': split,
                'index': idx
            }
        }
        return data
    return process_fn


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process datasets for DeepScaler training')
    parser.add_argument('--local_dir', default=os.path.expanduser('../raw_data/deepscaler'),
                       help='Local directory to save processed datasets')
    parser.add_argument('--train_dir', default=os.path.expanduser('../train'),
                       help='Local directory to save processed datasets')
    parser.add_argument('--val_dir', default=os.path.expanduser('../val'),
                       help='Local directory to save processed datasets')
    args = parser.parse_args()

    local_dir = args.local_dir
    train_dir = args.train_dir
    val_dir = args.val_dir
    # Make local directory if it doesn't exist
    makedirs(train_dir, exist_ok=True)
    makedirs(val_dir, exist_ok=True)

    # Initialize datasets
    dataset = load_dataset(os.path.join(local_dir,"deepscaler.json"))
    train_size = 32
    val_size = 500

    #randomly shuffle the dataset
    random.shuffle(dataset)
    sampled_dataset = dataset[:train_size + val_size]
    train_dataset = sampled_dataset[:train_size]
    val_dataset = sampled_dataset[train_size:]

    # Process training data
    train_data: List[Dict[str, Any]] = []
    val_data: List[Dict[str, Any]] = []
    process_fn = make_map_fn('train')
    for idx, example in enumerate(train_dataset):
        processed_example = process_fn(example, idx, "deepscaler")
        if processed_example is not None:
            train_data.append(processed_example)

    process_fn = make_map_fn('val')
    for idx, example in enumerate(val_dataset):
        processed_example = process_fn(example, idx, "deepscaler")
        if processed_example is not None:
            val_data.append(processed_example)

    # Save training dataset
    print("train data size:", len(train_data))
    train_df = pd.DataFrame(train_data)
    # repeat the training data to make it 128 samples
    train_df = pd.concat([train_df]*4, ignore_index=True).sample(frac=1).reset_index(drop=True)
    train_df.to_parquet(os.path.join(train_dir, 'deepscaler_train.parquet'))

    print(train_df.head())

    print("val data size:", len(val_data))
    train_df = pd.DataFrame(val_data)
    train_df.to_parquet(os.path.join(val_dir, 'deepscaler_val.parquet'))