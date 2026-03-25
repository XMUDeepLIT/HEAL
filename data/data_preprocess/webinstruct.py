import pandas as pd


def read_parquet(file_path):
    df = pd.read_parquet(file_path)
    print(f"Dataframe loaded from {file_path} with shape {df.shape}")
    print(df.head())
    print(df.columns)
    return df


def extrac_phy(df):
    df_phy = df[df['category'] == 'Physics']
    print(f"Extracted physics category with shape {df_phy.shape}")
    print(df_phy.head())
    return df_phy


def tranform_format(df, split='train'):
    cloumns = ['data_source', 'prompt', 'ability', 'reward_model', 'extra_info']
    system_prompt = "\nLet's think step by step and output the final answer within \\boxed{}."
    new_df = pd.DataFrame(columns=cloumns)
    for i, row in df.iterrows():
        new_row = {}
        new_row['data_source'] = 'webinstruct-phy'
        question = row['question']
        answer = row['answer']
        prompt = question + system_prompt
        new_row['prompt'] = [{
            'role': 'user',
            'content': prompt
        }]
        new_row['ability'] = f'{row["difficulty"]} physics'
        new_row['reward_model'] = {
            'ground_truth': answer,
            'style': 'rule'
        }
        index = f"webinstruct-phy_{row['answer_type']}_{row['id']}"
        new_row['extra_info'] = {'index': index, 'split': split}
        new_df = pd.concat([new_df, pd.DataFrame([new_row])], ignore_index=True)
    print(f"Transformed dataframe shape: {new_df.shape}")
    print(new_df.head())
    return new_df


if __name__ == "__main__":
    input_path = "../raw_data/webinstruct/data/train_legacy-00000-of-00001.parquet"
    train_path = "../train/webinstruct_phy_train.parquet"
    val_path = "../val/webinstruct_phy_val.parquet"
    df = read_parquet(input_path)
    df_phy = extrac_phy(df)

    train_size = 32
    val_size = 500
    new_df_phy = df_phy.sample(frac=1).reset_index(drop=True).head(train_size + val_size)
    train_df = new_df_phy.head(train_size)
    train_df = pd.concat([train_df]*4, ignore_index=True).sample(frac=1).reset_index(drop=True)
    val_df = new_df_phy.tail(val_size)

    print(f"Shuffled and sampled dataframe shape: {train_df.shape} and {val_df.shape}")
    train_df = tranform_format(train_df, "train")
    train_df.to_parquet(train_path)
    val_df = tranform_format(val_df, "val")
    val_df.to_parquet(val_path)
    print(f"Transformed dataframe saved to {train_path} and {val_path}")