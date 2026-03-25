# Copyright 2024 PRIME team and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Callable, Optional
import math
import psutil
import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizer

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register


async def single_compute_score(evaluation_func, completion, reference, task, task_extra_info, executor, timeout=15.0):
    loop = asyncio.get_running_loop()
    try:
        # Ensure process_completion is called properly
        future = loop.run_in_executor(executor, partial(evaluation_func, task, completion, reference, task_extra_info))
        return await asyncio.wait_for(future, timeout=timeout)
    except asyncio.TimeoutError:
        print(f"[Timeout] Task timeout: {task_extra_info}")
        return None  # Default value for timed-out rows
    except Exception as e:
        print(f"[Error] Task failed: {e}, completion: {completion[:80]}")
        return None  # Default value for failed rows


async def parallel_compute_score_async(
    evaluation_func, completions, references, tasks, extra_info=None, num_processes=64
):
    if extra_info is None:
        extra_info = [None] * len(tasks)
    scores = []
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # to prevent very occasional starvation caused by some anomalous programs ( like infinite loop ), the
        # exceptions in async programs will instantly halt the evaluation, and all summoned processes will be killed.
        try:
            # Create tasks for all rows
            tasks_async = [
                single_compute_score(evaluation_func, c, r, t, ei, executor, timeout=15.0)
                for c, r, t, ei in zip(completions, references, tasks, extra_info)
            ]
            results = await asyncio.gather(*tasks_async, return_exceptions=False)
        except Exception as e:
            print(f"[Exception] async gather failed: {e}")
            raise
        finally:
            terminated_count = 0
            for pid, proc in executor._processes.items():
                try:
                    p = psutil.Process(pid)
                    p.terminate()
                    try:
                        p.wait(timeout=5)
                    except psutil.TimeoutExpired:
                        p.kill()
                    terminated_count += 1
                except Exception:
                    pass
            print(f"[Shutdown] {terminated_count} subprocess(es) terminated.")

    # Process results
    for result, completion, reference, task in zip(results, completions, references, tasks):
        if isinstance(result, Exception) or result is None:
            # Handle failed or timed-out tasks
            scores.append(0.0)
        elif isinstance(result, (int, float, bool)):
            scores.append(float(result))
        else:
            scores.append(float(result[0]))
    return scores


def run_reward_scoring(evaluation_func, completions, references, tasks, extra_info=None, num_processes=64):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(
            parallel_compute_score_async(evaluation_func, completions, references, tasks, extra_info, num_processes)
        )
    finally:
        loop.close()


def compute_single_score(i, all_entropies, all_lengths, all_sources):
    """
    计算单个样本 i 的 rt_score
    """
    n = len(all_entropies)
    val_len_i = all_lengths[i]
    ds_i = all_sources[i]
    
    if val_len_i <= 50:
        return 0.0

    valid_token_entropy_i = all_entropies[i][:val_len_i]
    min_dt_gen = math.inf
    min_dt_tg = math.inf

    for j in range(n):
        if i == j:
            continue
        
        ds_j = all_sources[j]
        val_len_j = all_lengths[j]
        valid_token_entropy_j = all_entropies[j][:val_len_j]
        
        # 克隆以防修改原数据
        ent_i = valid_token_entropy_i.clone()
        ent_j = valid_token_entropy_j.clone()

        # 长度对齐
        if val_len_j != val_len_i:
            if val_len_i > val_len_j:
                ent_j = F.interpolate(ent_j.view(1, 1, -1), size=val_len_i, mode="nearest").view(-1)
            else:
                ent_i = F.interpolate(ent_i.view(1, 1, -1), size=val_len_j, mode="nearest").view(-1)

        # KL 散度计算
        # 注意：kl_div 的输入通常期望 log_probabilities
        kl = F.kl_div(ent_j.softmax(-1).log(), ent_i.softmax(-1), reduction='sum').item()

        if ds_j == "commonsense_qa":
            min_dt_gen = min(kl, min_dt_gen)
        else:
            min_dt_tg = min(kl, min_dt_tg)

    if ds_i == "commonsense_qa":
        return 0.0 if min_dt_tg > min_dt_gen else 1.0
    else:
        return 1.0 if min_dt_tg > min_dt_gen else 0.0


@register("prime_heal")
class HEALRewardManager:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        num_examine: int,
        compute_score: Optional[Callable] = None,
        reward_fn_key: str = "data_source",
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key

    def verify(self, data):
        """
        verify the batch and save as ``acc`` tensor
        """
        # batched scoring
        prompt_ids = data.batch["prompts"]

        response_ids = data.batch["responses"]
        sequences_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        ground_truth = [data_item.non_tensor_batch["reward_model"]["ground_truth"] for data_item in data]
        data_sources = data.non_tensor_batch[self.reward_fn_key]
        extra_info = data.non_tensor_batch.get("extra_info", None)

        assert len(sequences_str) == len(ground_truth) == len(data_sources)
        try:
            scores = run_reward_scoring(
                self.compute_score,
                completions=sequences_str,
                references=ground_truth,
                tasks=data_sources,
                extra_info=extra_info,
                num_processes=64,
            )
        except asyncio.TimeoutError:
            print("[Timeout] Global reward scoring timed out. Setting all as 0.")
            scores = [0.0 for _ in range(len(sequences_str))]
        except Exception as e:
            print(f"[Error] Unexpected error during scoring. Setting all as 0. {e}")
            scores = [0.0 for _ in range(len(sequences_str))]
        data.batch["acc"] = torch.tensor(scores, dtype=torch.float32, device=prompt_ids.device)
        return scores


    def eda_score(self, data):
        """
        多进程版本的 rt_kl_score
        """
        prompt_ids = data.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        token_entropy = data.batch["token_entropys"].cpu() # 传给多进程前建议先转到 CPU
        valid_response_length = data.batch["attention_mask"][:, prompt_length:].sum(dim=-1).cpu().tolist()
        data_sources = data.non_tensor_batch["data_source"]
        
        num_samples = len(data)
        
        # 准备并行任务
        # 将 Tensor 转换为列表可以避免在某些系统下多进程传递 Tensor 的序列化问题
        # 如果内存允许，直接传递整个 Tensor 块更高效
        indices = list(range(num_samples))
        
        # 使用偏函数绑定共有数据
        worker_func = partial(
            compute_single_score, 
            all_entropies=token_entropy, 
            all_lengths=valid_response_length, 
            all_sources=data_sources
        )

        max_workers = 128
        # print(f"Starting parallel RT score calculation with {max_workers} workers...")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 使用 map 保持顺序
            rt_scores = list(executor.map(worker_func, indices))
        
        data.batch["rt_scores"] = torch.tensor(rt_scores, dtype=torch.float32, device=prompt_ids.device)
        return rt_scores


    def __call__(self, data: DataProto, return_dict: bool = False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:   
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

        already_print_data_sources = {}

        # batched scoring
        prompt_ids = data.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]

        response_ids = data.batch["responses"]
        valid_response_length = data.batch["attention_mask"][:, prompt_length:].sum(dim=-1)
        sequences_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        ground_truth = [data_item.non_tensor_batch["reward_model"]["ground_truth"] for data_item in data]
        data_sources = data.non_tensor_batch["data_source"]
        data_idx = data.non_tensor_batch["index"]
        if "rollout_log_probs" in data.batch.keys():
            token_entropy = data.batch["token_entropys"]
        else:
            token_entropy = None

        scores = self.verify(data)
        eda_scores = self.eda_score(data)

        final_scores = [s+rt for s,rt in zip(scores,eda_scores)]
        rollouts_info = []


        for i in range(len(data)):
            data_source = data_sources[i]
            reward_tensor[i, valid_response_length[i].item() - 1] = final_scores[i]
            valid_token_entropy = token_entropy[i, : valid_response_length[i].item()]
            if data_source in  ["codecontests", "apps", "codeforces", "taco"]:
                ground_truth_i = None
            else:
                ground_truth_i = ground_truth[i]
            rollouts_info_dict = {
                "data_source": data_source,
                "ground_truth": ground_truth_i,
                "response": sequences_str[i],
                "score": scores[i],
                "eda_score": eda_scores[i],
                "data_idx": data_idx[i],
                "token_entropy": valid_token_entropy.numpy().tolist(),
            }
            rollouts_info.append(rollouts_info_dict)
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)

        if return_dict:
            res_dict = {"reward_tensor": reward_tensor, "rollouts_info": rollouts_info}
            return res_dict
        else:
            return reward_tensor
