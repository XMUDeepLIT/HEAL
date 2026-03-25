import pandas as pd
import os
from verl.utils.hdfs_io import makedirs

'''[{'content': "The pressure \\( P \\) exerted by wind on a sail varies jointly as the area \\( A \\) of the sail and the cube of the wind's velocity \\( V \\). When the velocity is \\( 8 \\) miles per hour, the pressure on a sail of \\( 2 \\) square feet is \\( 4 \\) pounds. Find the wind velocity when the pressure on \\( 4 \\) square feet of sail is \\( 32 \\) pounds. Let's think step by step and output the final answer within \\boxed{}.", 'role': 'user'}]'''


cloumns = ['data_source', 'prompt', 'ability', 'reward_model', 'extra_info']
tmp_output_file = "commonsense_qa_train.parquet"
raw_input_file = "../raw_data/commonsense_qa/data/train-00000-of-00001.parquet"
tmp_dir = "tmp"
train_dir = "../train"
val_dir = "../val"
train_file_name = "commonsense_qa_train.parquet"
val_file_name = "commonsense_qa_val.parquet"
system_prompt = "\nLet's think step by step and output the final answer within \\boxed{}."



data_source = "commonsense_qa"
ablity = "commonsense_qa"
split = "train"


def get_prompt(question, choices):
    prompt = question + "\nAnswer Choices: "
    alpha_num = ['A', 'B', 'C', 'D', 'E']
    for i, choice in enumerate(choices):
        prompt += f"({alpha_num[i]}) {choice} "
    prompt += system_prompt
    return prompt


def process_data(raw_df):
    new_df = pd.DataFrame(columns=cloumns)
    for index, row in raw_df.iterrows(): 
        new_row = {}
        new_row['data_source'] = data_source
        question = row['question']
        if "Figure" in question:
            continue
        choices = row["choices"]["text"]
        prompt = get_prompt(question, choices)
        new_row['prompt'] = [{
            'role': 'user',
            'content': prompt
        }]
        new_row['ability'] = ablity
        new_row['reward_model'] = {
            'ground_truth': row['answerKey'],
            'style': 'rule'
        }
        new_row['extra_info'] = {'index': f"{data_source}_{index}", 'split': split}
        # append to new_df
        new_df = pd.concat([new_df, pd.DataFrame([new_row])], ignore_index=True)
    print(new_df.head())
    print(new_df.shape)
    print(new_df['prompt'][0])
    tmp_output_file_path = os.path.join(tmp_dir, tmp_output_file)
    new_df.to_parquet(tmp_output_file_path, engine='pyarrow', index=False)
    print(f"Processed data saved to {tmp_output_file}")
    return new_df


def split_data(df, train_file, val_file, train_size=384, val_size=500):
    df = df.sample(frac=1).reset_index(drop=True)
    train_df = df.head(train_size)
    val_df = df.tail(val_size)
    print(f"Train dataframe shape: {train_df.shape}, Val dataframe shape: {val_df.shape}")
    print(train_df.head())
    train_df.to_parquet(train_file, engine='pyarrow', index=False)
    val_df.to_parquet(val_file, engine='pyarrow', index=False)
    print(f"Train and Val data saved to {train_file} and {val_file}")


if __name__ == "__main__":
    makedirs(tmp_dir, exist_ok=True)
    raw_df = pd.read_parquet(raw_input_file, engine='pyarrow')
    processed_df = process_data(raw_df)

    # split data into train and val
    train_file = os.path.join(train_dir, train_file_name)
    val_file = os.path.join(val_dir, val_file_name)
    split_data(processed_df, train_file, val_file, train_size=384, val_size=500)

