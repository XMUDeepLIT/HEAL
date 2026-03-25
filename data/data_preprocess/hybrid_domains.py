import pandas as pd
import os

train_dir = "../train"

target_domain_file_paths = [
    "deepscaler_train.parquet",
    "medbul_xqa_train.parquet",
    "webinstruct_phy_train.parquet",
    "livecode_train.parquet"
]

target_domain_file_output_paths = [
    "hybrid_deepscaler_train.parquet",
    "hybrid_medbul_xqa_train.parquet",
    "hybrid_webinstruct_phy_train.parquet",
    "hybrid_livecode_train.parquet"
]

general_domain_file_path = "commonsense_qa_train.parquet"

# 定义切分函数
def split_dataframe(df, n):
    import math
    df_num = len(df)
    every_epoch_num = math.floor(df_num / n)
    dfs = []
    for i in range(n):
        if i < n - 1:
            dfs.append(df[i * every_epoch_num: (i + 1) * every_epoch_num])
        else:
            dfs.append(df[i * every_epoch_num:])    
    return dfs

def merge_parquet(file_path_list, save_paths):
    n = 4

    general_domain_file = os.path.join(train_dir, general_domain_file_path) 
    gen_df = pd.read_parquet(general_domain_file, engine='pyarrow')
    gen_dfs = split_dataframe(gen_df,n) 
    for file_path, save_path in zip(file_path_list, save_paths):
        file_path = os.path.join(train_dir, file_path)
        df = pd.read_parquet(file_path)
        print(f"Loaded {file_path} with shape {df.shape}")
        for i, row in df.iterrows():
            index = row['extra_info']['index']
            row['extra_info']['index'] = str(index)
        data_frames = split_dataframe(df,n)
        for i,d in enumerate(data_frames):
            cdf = pd.concat([d,gen_dfs[i]], ignore_index=True)
            cdf = cdf.sample(frac=1).reset_index(drop=True)
            data_frames[i] = cdf
        combined_df = pd.concat(data_frames, ignore_index=True)
        print(f"Combined dataframe shape: {combined_df.shape}")
        print(combined_df.head(128))
        save_file_path = os.path.join(train_dir, save_path)
        combined_df.to_parquet(save_file_path)
        print(f"Combined dataframe saved to {save_file_path}")


if __name__ == "__main__":
    merge_parquet(target_domain_file_paths, target_domain_file_output_paths)