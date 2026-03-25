import pandas as pd
import jsonlines
import os
from verl.utils.hdfs_io import makedirs

'''[{'content': "The pressure \\( P \\) exerted by wind on a sail varies jointly as the area \\( A \\) of the sail and the cube of the wind's velocity \\( V \\). When the velocity is \\( 8 \\) miles per hour, the pressure on a sail of \\( 2 \\) square feet is \\( 4 \\) pounds. Find the wind velocity when the pressure on \\( 4 \\) square feet of sail is \\( 32 \\) pounds. Let's think step by step and output the final answer within \\boxed{}.", 'role': 'user'}]'''


cloumns = ['data_source', 'prompt', 'ability', 'reward_model', 'extra_info']
tmp_medbullets_output_file = "medbullets_trian.parquet"
raw_medbullets_input_files = ["../raw_data/medbullets/data/op4_test-00000-of-00001.parquet","../raw_data/medbullets/data/op5_test-00000-of-00001.parquet"]
tmp_medxqa_output_file = "medxqa_train.parquet"
raw_medxqa_input_file = "../raw_data/MedXpertQA/Text/test.jsonl"
tmp_dir = "tmp"
train_dir = "../train"
val_dir = "../val"
train_file_name = "medbul_xqa_train.parquet"
val_file_name = "medbul_xqa_val.parquet"
system_prompt = "\nLet's think step by step and output the final answer within \\boxed{}."



data_source = "medbullets"
MedXpertQA_data_source="MedXpertQA"
ablity = "medicine"
split = "train"


def read_medbullets_parquet(input_files):
    for i, file in enumerate(input_files):
        if i == 0:
            df = pd.read_parquet(file, engine='pyarrow')
        else:
            new_df = pd.read_parquet(file, engine='pyarrow')
            df = pd.concat([df, new_df], ignore_index=True)
    return df


def get_medbullets_prompt(question, choices):
    prompt = question + "\nAnswer Choices: "
    alpha_num = ['A', 'B', 'C', 'D', 'E']
    for i, choice in enumerate(choices):
        prompt += f"({alpha_num[i]}) {choice} "
    prompt += system_prompt
    return prompt


def process_medxqa_data(input_file):
    data = []
    with jsonlines.open(input_file) as reader:
        for obj in reader:
            data.append(obj)
    print(f"Loaded {len(data)} entries from {input_file}")
    new_df = pd.DataFrame(columns=cloumns)
    for i, row in enumerate(data):
        new_row = {}
        new_row['data_source'] = MedXpertQA_data_source
        question = row['question'] + system_prompt
        new_row['prompt'] = [{
            'role': 'user',
            'content': question
        }]
        new_row['ability'] = ablity
        new_row['reward_model'] = {
            'ground_truth': row["label"],
            'style': 'rule'
        }
        index = f"{MedXpertQA_data_source}_{row['id']}"
        
        new_row['extra_info'] = {'index': index, 'split': split}
        new_df = pd.concat([new_df, pd.DataFrame([new_row])], ignore_index=True)
    print(new_df.head())
    print(new_df.shape)
    print(new_df['prompt'][0])
    tmp_medxqa_output_file_path = os.path.join(tmp_dir, tmp_medxqa_output_file)
    new_df.to_parquet(tmp_medxqa_output_file_path, engine='pyarrow', index=False)
    print(f"Processed data saved to {tmp_medxqa_output_file}")
    return new_df


def process_medbullets_data(raw_df):
    new_df = pd.DataFrame(columns=cloumns)
    for index, row in raw_df.iterrows(): 
        new_row = {}
        new_row['data_source'] = data_source
        question = row['question']
        if "Figure" in question:
            continue
        choices = [row['options']['A'], row['options']['B'], row['options']['C'], row['options']['D']]
        if row['options']['E'] is not None and row['options']['E'] != '':
            choices.append(row['options']['E'])
        prompt = get_medbullets_prompt(question, choices)
        new_row['prompt'] = [{
            'role': 'user',
            'content': prompt
        }]
        new_row['ability'] = ablity
        new_row['reward_model'] = {
            'ground_truth': row['answer'],
            'style': 'rule'
        }
        new_row['extra_info'] = {'index': f"{data_source}_{index}", 'split': split}
        # append to new_df
        new_df = pd.concat([new_df, pd.DataFrame([new_row])], ignore_index=True)
    print(new_df.head())
    print(new_df.shape)
    print(new_df['prompt'][0])
    tmp_medbullets_output_file_path = os.path.join(tmp_dir, tmp_medbullets_output_file)
    new_df.to_parquet(tmp_medbullets_output_file_path, engine='pyarrow', index=False)
    print(f"Processed data saved to {tmp_medbullets_output_file}")
    return new_df


def split_data(combined_df, train_file, val_file, train_size=32, val_size=500):
    df = combined_df
    df = df.sample(frac=1).reset_index(drop=True)
    train_df = df.head(train_size)
    train_df = pd.concat([train_df]*4, ignore_index=True).sample(frac=1).reset_index(drop=True)
    val_df = df.tail(val_size)
    print(f"Train dataframe shape: {train_df.shape}, Val dataframe shape: {val_df.shape}")
    print(train_df.head())
    train_df.to_parquet(train_file, engine='pyarrow', index=False)
    val_df.to_parquet(val_file, engine='pyarrow', index=False)
    print(f"Train and Val data saved to {train_file} and {val_file}")


if __name__ == "__main__":
    makedirs(tmp_dir, exist_ok=True)
    raw_medbul_df = read_medbullets_parquet(raw_medbullets_input_files)
    processed_medbul_df = process_medbullets_data(raw_medbul_df)
    process_medxqa_df = process_medxqa_data(raw_medxqa_input_file)
    combined_df = pd.concat([processed_medbul_df, process_medxqa_df], ignore_index=True)
    # split data into train and val
    train_file = os.path.join(train_dir, train_file_name)
    val_file = os.path.join(val_dir, val_file_name)
    split_data(combined_df, train_file, val_file, train_size=32, val_size=500)

