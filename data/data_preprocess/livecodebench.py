import jsonlines
import pandas as pd
import json
import os
from verl.utils.hdfs_io import makedirs


raw_input_files = ['test.jsonl', 'test2.jsonl','test3.jsonl', 'test4.jsonl']
tmp_output_files = ['extracted_code.jsonl','extracted_code2.jsonl', 'extracted_code3.jsonl', 'extracted_code4.jsonl']
system_prompt = "\nLet's think step by step and Finally Write Python code to solve the problem. Present the code in \n```python\nYour code\n```\nYou need to read input and print output in Your code."
tmp_output_file = 'livecode_v1to4.parquet'
'''[{'content': "The pressure \\( P \\) exerted by wind on a sail varies jointly as the area \\( A \\) of the sail and the cube of the wind's velocity \\( V \\). When the velocity is \\( 8 \\) miles per hour, the pressure on a sail of \\( 2 \\) square feet is \\( 4 \\) pounds. Find the wind velocity when the pressure on \\( 4 \\) square feet of sail is \\( 32 \\) pounds. Let's think step by step and output the final answer within \\boxed{}.", 'role': 'user'}]'''


def extract_code(output_file, input_file):
    with jsonlines.open(input_file) as reader, jsonlines.open(output_file, mode='w') as writer:
        data = []
        for obj in reader:
            new_obj = {}
            new_obj["question_content"] = obj.get('question_content', '')
            new_obj["platform"] = obj.get('platform', '')
            new_obj["question_id"] = obj.get('question_id', '')
            new_obj["starter_code"] = obj.get('starter_code', '')
            new_obj['difficulty'] = obj.get('difficulty', '')
            new_obj['public_test_cases'] = obj.get('public_test_cases', [])
            data.append(new_obj)
        print(f"Extracted {len(data)} entries.")
        writer.write_all(data)
    print(f"Extracted code saved to {output_file}")


def get_answer(public_test_cases):
    '''"ground_truth": "{\"fn_name\": \"findNumber\", \"inputs\": [[1], [5], [10], [9], [7], [1], [2], [3], [4], [11], [1000000000000],[999999999999]], \"outputs\": [[\"1\"], [\"9\"], [\"19\"], [\"17\"], [\"13\"], [\"1\"], [\"3\"], [\"5\"], [\"7\"], [\"31\"], [\"113559777777777779\"], [\"113559777777777777\"]]}","style": "rule"'''    
    inputs = []
    outputs = []
    # transfer str to dict
    public_test_cases = json.loads(public_test_cases)
    for case in public_test_cases:
        inputs.append(case.get('input', ''))
        outputs.append(case.get('output', ''))
    ground_truth = {
                "inputs": inputs,
                "outputs": outputs
    }
    # to str
    ground_truth = json.dumps(ground_truth)
    return {
            "ground_truth": ground_truth,
            "style": "rule"
        }


def process_df(tmp_dir, tmp_output_files, tmp_parquet_file):
    data = []
    for input_file_name in tmp_output_files:
        input_file = os.path.join(tmp_dir, input_file_name)
        with jsonlines.open(input_file) as reader:
            for obj in reader:
                new_obj = {}
                question_content = obj.get('question_content', '')
                platform = obj.get('platform', '')
                question_id = obj.get('question_id', '')
                difficulty = obj.get('difficulty', '')
                public_test_cases = obj.get('public_test_cases', [])
                prompt = question_content + system_prompt
                new_obj['data_source'] = platform
                new_obj['prompt'] = [{'content': prompt, 'role': 'user'}]
                ability = difficulty
                new_obj['ability'] = ability
                new_obj['reward_model'] = get_answer(public_test_cases)
                index = f"{platform}_{difficulty}_{question_id}"
                new_obj['extra_info'] = {'index': index, 'split': 'train'}
                data.append(new_obj)

        # with jsonlines.open("livecodetrain.jsonl", mode='w') as writer:
        #     writer.write_all(data)
        print(f"Processed {len(data)} entries.")
        df = pd.DataFrame(data)
        print(df.head())
        print(df.shape)
        print(df['prompt'][0])
        print(df["reward_model"][0])
        df.to_parquet(tmp_parquet_file, engine='pyarrow', index=False)
        print(f"Processed data saved to livecodetrain.parquet")


def split_data(input_parquet_file, train_file, val_file, train_size=32, val_size=500):
    df = pd.read_parquet(input_parquet_file)
    print(f"Dataframe loaded from {input_parquet_file} with shape {df.shape}")
    df = df.sample(frac=1).reset_index(drop=True)
    train_df = df.head(train_size)
    train_df = pd.concat([train_df]*4, ignore_index=True).sample(frac=1).reset_index(drop=True)
    val_df = df.tail(val_size)
    print(f"Train dataframe shape: {train_df.shape}, Val dataframe shape: {val_df.shape}")
    print(train_df.head())
    train_df.to_parquet(train_file, engine='pyarrow', index=False)

    val_df.to_parquet(val_file, engine='pyarrow', index=False)
    print(f"Train and Val data saved to livecodetrain.parquet and livecodeval.parquet")


if __name__ == "__main__":
    tmp_dir = "tmp"
    train_dir = "../train"
    val_dir = "../val"
    raw_dir = "../raw_data/livecodebench"
    makedirs(tmp_dir, exist_ok=True)

    for raw_file, tmp_file in zip(raw_input_files, tmp_output_files):
        input_file = os.path.join(raw_dir, raw_file)
        output_file = os.path.join(tmp_dir, tmp_file)
        extract_code(output_file, input_file)

    tmp_parquet_file = os.path.join(tmp_dir, tmp_output_file)
    process_df(tmp_dir, tmp_output_files, tmp_parquet_file)

    train_file = os.path.join(train_dir, "livecode_train.parquet")
    val_file = os.path.join(val_dir, "livecode_val.parquet")
    split_data(tmp_parquet_file, train_file, val_file, train_size=32, val_size=500)