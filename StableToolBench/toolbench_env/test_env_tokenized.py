import json
import os
import time
from dataclasses import dataclass
from typing import List

# ==========================================
from build_env import build_toolbench_envs
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoTokenizer

# ==========================================

# ================= 配置区域 =================
ENV_NUM = 163
GROUP_N = 1  # 比如这里设为3，表示每个Query采样3次
SEED = 42
MAX_SAMPLES = 163  # 指定测试样本数量，None 表示跑完全部数据；设为整数则达到该数量后停止

os.environ["NO_PROXY"] = "localhost,127.0.0.1"
VLLM_API_URL = "http://127.0.0.1:8000/v1"
#VLLM_MODEL_NAME = "Qwen/Qwen3-8B"
#MODEL_PATH = "Qwen/Qwen3-8B"
#VLLM_MODEL_NAME = "ToolBench/ToolLLaMA-2-7b-v2"

VLLM_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_PATH = "Qwen/Qwen3-4B-Instruct-2507"

STOP_TOKENS = ["<|im_end|>", "<|endoftext|>", "</s>"]


@dataclass
class SpecificArgs:
    input_query_dir: str = "solvable_queries/test_instruction/G1_instruction.json"
    corpus_tsv_path: str = None
    retrieval_model_path: str = None
    max_sequence_length: int = 1024
    evaluator_name: str = "tooleval_aliyun-deepseek-normalization"
    evaluators_cfg_path: str = "toolbench/tooleval/evaluators"
    #template: str = "chat_model"
    template: str = "tool-llama-single-round"
    single_chain_max_step: int = 12
    evaluation_times: int = 1
    model_path: str = MODEL_PATH
    tool_root_dir: str = "tools_folder/server_cache/tools"
    toolbench_key: str = "EMPTY"
    rapidapi_key: str = "EMPTY"
    use_rapidapi_key: bool = False
    api_customization: bool = False
    use_retriever: bool = False
    max_observation_length: int = 1024
    observ_compress_method: str = "truncate"
    base_url: str = VLLM_API_URL
    method: str = "CoT"
    tree_beam_size: int = 4
    max_query_count: int = 200
    answer: int = 1


# ===========================================


def get_vllm_client():
    return OpenAI(base_url=VLLM_API_URL, api_key="EMPTY")


def get_tokenizer():
    try:
        print(f"⏳ Loading Tokenizer: {MODEL_PATH}...")
        # 确保加载 tokenizer，这对于本地 tokenization 是必须的
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        print("✅ Tokenizer loaded.")
        return tokenizer
    except Exception as e:
        print(f"❌ Tokenizer load failed: {e}")
        return None


def generate_actions_openai_batch(
    client: OpenAI,
    tokenizer: AutoTokenizer,
    observations: List[str],
    dones: List[bool],
    args: SpecificArgs,
) -> List[str]:
    """生成动作 Batch"""
    total = len(observations)
    actions = [None] * total
    active_indices = []
    active_prompts = []

    # 筛选出需要生成的 prompt
    for i, (obs, done) in enumerate(zip(observations, dones)):
        if done:
            actions[i] = "FINISHED"
        elif not obs:
            actions[i] = ""
        else:
            active_indices.append(i)
            active_prompts.append(obs)

    if not active_prompts:
        return actions

    try:
        prompt_inputs = active_prompts

        # ============ 修改开始: Tokenize ============
        if tokenizer:
            # 使用 tokenizer.encode 将文本转换为 token id 列表
            # 结果形式为 [[id1, id2, ...], [id1, id2, ...]]
            # 注意：vLLM/OpenAI 接口支持 prompt 参数为 List[List[int]]
            prompt_inputs = [tokenizer.encode(p, add_special_tokens=True) for p in active_prompts]
        else:
            print("⚠️ Warning: Tokenizer is missing, falling back to raw text string.")
        # ============ 修改结束 ============
        for i, prompt_input in enumerate(prompt_inputs):
            if len(prompt_input) > 2 * 4096 - args.max_sequence_length:
                prompt_inputs[i] = prompt_input[-(2 * 4096 - args.max_sequence_length) :]

        """
        response = client.completions.create(
            model=VLLM_MODEL_NAME,
            prompt=prompt_inputs,  # 这里传入的是 token ids 列表（如果 tokenizer 存在）
            max_tokens=args.max_sequence_length,
            temperature=1.0,
            # seed=SEED,  # 与 test_vllm_direct 端相同 seed，保证采样一致、消除 reward 差距
            # stop=STOP_TOKENS,
            extra_body={
                "truncate_prompt_tokens": 4096,  # 如果 token 仍然过长，vLLM 会根据此参数截断
            },
        )
        """
        extra_body = {
            "truncate_prompt_tokens": 2 * 4096 - args.max_sequence_length,
            # "top_p": 0.8,
            # "top_k": 20,
            # "repetition_penalty": 1.05,
            # "frequency_penalty": 0.0,
        }
        if getattr(tokenizer, "eos_token_id", None) is not None:
            extra_body["stop_token_ids"] = [tokenizer.eos_token_id]
        response = client.completions.create(
            model=VLLM_MODEL_NAME,
            prompt=prompt_inputs,
            max_tokens=args.max_sequence_length,
            temperature=1.0,
            n=1,
            logprobs=0,
            extra_body=extra_body,
        )

        # 逐个打印实际prompt长度
        print(f"实际prompt长度: {response.usage.prompt_tokens}")

        choices = response.choices
        for i, choice in enumerate(choices):
            if i < len(active_indices):
                original_idx = active_indices[i]
                actions[original_idx] = choice.text.strip()

    except Exception as e:
        print(f"❌ LLM Request Failed: {e}")
        for idx in active_indices:
            if actions[idx] is None:
                actions[idx] = ""

    for i in range(total):
        if actions[i] is None:
            actions[i] = ""

    return actions


def main():
    args = SpecificArgs()

    if not os.path.exists(args.input_query_dir):
        print(f"❌ Input file not found: {args.input_query_dir}")
        return

    client = get_vllm_client()
    tokenizer = get_tokenizer()

    if tokenizer is None:
        print("❌ 必须成功加载 Tokenizer 才能进行本地 Tokenize 操作，程序退出。")
        return

    try:
        envs = build_toolbench_envs(
            env_num=ENV_NUM,
            group_n=GROUP_N,
            resources_per_worker={"num_cpus": 100},
            is_train=False,
            specific_args=args,
        )
    except Exception as e:
        print(f"❌ 环境构建失败: {e}")
        return

    total_dataset_len = len(envs._query_list)

    print("🚀 初始化 ToolBench 全量评测 (With Client-Side Tokenization)")
    print(f"Dataset Size: {total_dataset_len} | Batch Size: {ENV_NUM} | Group N: {GROUP_N}")

    # ================= 状态追踪变量 =================
    group_stats = [[] for _ in range(GROUP_N)]
    global_processed_count = 0
    global_final_rewards = []

    pbar_total = min(total_dataset_len, MAX_SAMPLES) if MAX_SAMPLES is not None else total_dataset_len
    pbar = tqdm(total=pbar_total, desc="Total Progress", unit="sample")

    while True:
        if global_processed_count >= MAX_SAMPLES:
            print(f"✅ 已达到指定样本数量 {MAX_SAMPLES}，停止测试。")
            break
        observations, infos = envs.reset()

        current_batch_size = len(observations)

        # 计算当前 Batch 每个任务的 Group ID
        batch_group_ids = []
        for i in range(current_batch_size):
            abs_index = global_processed_count + i
            group_id = abs_index % GROUP_N
            batch_group_ids.append(group_id)

        global_processed_count += current_batch_size

        dones = [False] * current_batch_size
        episode_rewards = [0.0] * current_batch_size
        final_rewards = [None] * current_batch_size

        step_cnt = 0
        max_steps = 15

        while not all(dones) and step_cnt < max_steps:
            step_cnt += 1
            actions = generate_actions_openai_batch(client, tokenizer, observations, dones, args)
            next_obs, step_rewards, next_dones, step_infos = envs.step(actions)
            # import pdb; pdb.set_trace()

            for i in range(current_batch_size):
                if not dones[i]:
                    episode_rewards[i] += step_rewards[i]
                    if next_dones[i]:
                        final_rewards[i] = episode_rewards[i]
                        dones[i] = True
                    observations[i] = next_obs[i]
            # print(f"Step:{step_cnt}") # 减少打印刷屏

        for i in range(current_batch_size):
            reward = final_rewards[i] if final_rewards[i] is not None else 0.0
            global_final_rewards.append(reward)

            g_id = batch_group_ids[i]
            group_stats[g_id].append(reward)

        pbar.update(current_batch_size)
        curr_avg = sum(global_final_rewards) / len(global_final_rewards) if global_final_rewards else 0
        pbar.set_postfix({"Avg": f"{curr_avg:.3f}"})

    pbar.close()

    # ================= 最终报告 =================
    print("\n" + "=" * 60)
    print("📊 Final Evaluation Report (Grouped)")
    print("=" * 60)

    total_evaluated = len(global_final_rewards)
    overall_avg = sum(global_final_rewards) / total_evaluated if total_evaluated > 0 else 0.0

    print(f"Total Instances Processed: {total_evaluated}")
    print(f"Overall Average Reward   : {overall_avg:.4f}")
    print("-" * 60)
    print(f"{'Group ID':<10} | {'Count':<10} | {'Avg Reward':<12} | {'Pass Rate':<12}")
    print("-" * 60)

    group_results_json = {}

    for g_id in range(GROUP_N):
        rewards = group_stats[g_id]
        count = len(rewards)
        if count > 0:
            avg_r = sum(rewards) / count
            pass_r = sum(1 for r in rewards if r >= 1.0) / count
        else:
            avg_r = 0.0
            pass_r = 0.0

        print(f"Sample {g_id:<3} | {count:<10} | {avg_r:.4f}       | {pass_r:.2%}")

        group_results_json[f"sample_{g_id}"] = {
            "count": count,
            "average_reward": avg_r,
            "pass_rate": pass_r,
            "rewards": rewards,
        }

    result_file = f"eval_result_grouped_{int(time.time())}.json"
    with open(result_file, "w") as f:
        json.dump(
            {
                "total_evaluated": total_evaluated,
                "overall_average": overall_avg,
                "group_n": GROUP_N,
                "group_details": group_results_json,
                "all_rewards_flat": global_final_rewards,
            },
            f,
            indent=4,
        )
    print(f"\n📝 详细分组结果已保存至: {result_file}")


if __name__ == "__main__":
    main()
