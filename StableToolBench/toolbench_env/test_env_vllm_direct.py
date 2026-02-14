"""
与 test_single_chain_env_tokenized.py 逻辑一致，但将「调用 serve API」改为「vLLM 进程内直接 generate」。
参数与生成逻辑与 vllm_rollout_spmd 保持一致，能复用的直接复用。
"""

# 必须在 import vllm 之前设置，否则会触发 "Cannot re-initialize CUDA in forked subprocess"
import os

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import json
import time
from dataclasses import dataclass
from typing import List, Optional

# ==========================================
from build_env import build_toolbench_envs
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoTokenizer

# 与 vllm_rollout_spmd 一致：复用 verl 工具
from verl.utils.torch_functional import pad_2d_list_to_length

# ================= 配置区域 =================
ENV_NUM = 100
GROUP_N = 1
SEED = 42
MAX_SAMPLES = 200

# 是否收集每个 episode 的 observation 轨迹并写入结果 JSON
COLLECT_OBSERVATIONS = True
# 写入 JSON 时每个 observation 字符串的最大长度，避免文件过大；None 表示不截断
MAX_OBS_LENGTH_SAVED = 2048

os.environ["NO_PROXY"] = "localhost,127.0.0.1"
# MODEL_PATH = "/home/u-longyy/efficient-verl-agent/checkpoints/verl_agent_toolbench_eval/grpo_qwen2.5_7b_toolbench/global_step_50/actor"
MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"

TRUNCATE_PROMPT_TOKENS = 7072

# 与 vllm_rollout_spmd 使用的 config 结构一致（eval.sh + ppo_trainer.yaml）
# 用于与 vllMRollout 相同的 kwargs 构建和 pad_2d_list_to_length
ROLLOUT_CONFIG = OmegaConf.create(
    {
        "prompt_length": 7072,
        "response_length": 1024,
        "temperature": 1.0,
        "top_k": -1,
        "top_p": 1,
        "n": 1,
        "logprobs": 0,
        "do_sample": True,
        "enforce_eager": True,
        "dtype": "bfloat16",
        "gpu_memory_utilization": 0.85,
        "max_num_batched_tokens": 15000,
        "max_num_seqs": 200,
        "load_format": "safetensors",
        "disable_log_stats": True,
        "enable_chunked_prefill": False,
        "enable_prefix_caching": True,
        "seed": 0,
        "val_kwargs": {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "n": 1,
        },
    }
)


@dataclass
class SpecificArgs:
    input_query_dir: str = "data/test_instruction/G1_instruction.json"
    corpus_tsv_path: str = None
    retrieval_model_path: str = None
    max_sequence_length: int = 1024
    evaluator_name: str = "tooleval_deepseek-normalization"
    evaluators_cfg_path: str = "toolbench/tooleval/evaluators"
    template: str = "chat_model"
    single_chain_max_step: int = 10
    evaluation_times: int = 1
    model_path: str = MODEL_PATH
    tool_root_dir: str = "data/toolenv/tools/"
    toolbench_key: str = "EMPTY"
    rapidapi_key: str = "EMPTY"
    use_rapidapi_key: bool = False
    api_customization: bool = False
    use_retriever: bool = False
    max_observation_length: int = 1024
    observ_compress_method: str = "truncate"
    base_url: str = "http://127.0.0.1:8000/v1"  # 仅用于 env 内部可选逻辑，本脚本不调 serve
    method: str = "DFS"
    tree_beam_size: int = 2
    max_query_count: int = 200
    answer: int = 1


# ===========================================


def get_vllm_llm_and_tokenizer():
    """进程内加载 vLLM 与 tokenizer，参数与 vllm_rollout_spmd 一致"""
    try:
        from vllm import LLM
    except ImportError as e:
        print(f"❌ 未安装 vllm: {e}")
        return None, None

    config = ROLLOUT_CONFIG
    max_model_len = config.prompt_length + config.response_length

    try:
        print(f"⏳ Loading vLLM model: {MODEL_PATH}...")
        load_format = "dummy" if str(config.load_format).startswith("dummy") else config.load_format
        llm = LLM(
            model=MODEL_PATH,
            trust_remote_code=True,
            max_model_len=max_model_len,
            gpu_memory_utilization=config.gpu_memory_utilization,
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            load_format=load_format,
            max_num_batched_tokens=config.max_num_batched_tokens,
            max_num_seqs=config.max_num_seqs,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=config.enable_prefix_caching,
            disable_log_stats=config.disable_log_stats,
            seed=config.get("seed", 0),
        )
        tokenizer = llm.get_tokenizer()
        print("✅ vLLM and tokenizer loaded.")
        return llm, tokenizer
    except Exception as e:
        print(f"❌ vLLM/Tokenizer load failed: {e}")
        return None, None


def get_hf_tokenizer():
    """与 tokenized 脚本一致：用 HF AutoTokenizer，保证 prompt encode 与 client 端完全相同"""
    try:
        print(f"⏳ Loading HF Tokenizer: {MODEL_PATH}...")
        tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        print("✅ HF Tokenizer loaded.")
        return tok
    except Exception as e:
        print(f"❌ HF Tokenizer load failed: {e}")
        return None


def generate_actions_vllm_direct(
    llm,
    tokenizer,
    hf_tokenizer,
    observations: List[str],
    dones: List[bool],
    args: SpecificArgs,
) -> List[str]:
    """vLLM 进程内 generate，SamplingParams 与 response 处理与 vllm_rollout_spmd 一致。"""
    from vllm import SamplingParams

    total = len(observations)
    actions = [None] * total
    active_indices = []
    active_prompts = []

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

    config = ROLLOUT_CONFIG
    # 与 tokenized 对齐：用 HF tokenizer 的 eos/pad 与 decode，便于和 client 端行为一致
    enc_dec_tok = hf_tokenizer if hf_tokenizer is not None else tokenizer
    pad_token_id = enc_dec_tok.pad_token_id
    print(f"pad_token_id: {pad_token_id}")
    eos_token_id = enc_dec_tok.eos_token_id
    print(f"eos_token_id: {eos_token_id}")
    pad_id = pad_token_id if pad_token_id is not None else eos_token_id

    try:
        # prompt_token_ids：与 tokenized 完全一致，用 HF tokenizer encode（client 端发来的就是 HF 编码）
        prompt_token_ids_list = []
        for p in active_prompts:
            ids = (hf_tokenizer if hf_tokenizer is not None else tokenizer).encode(p, add_special_tokens=True)
            if len(ids) > TRUNCATE_PROMPT_TOKENS:
                print(f"Truncating prompt from {len(ids)} to {TRUNCATE_PROMPT_TOKENS}")
                ids = ids[-TRUNCATE_PROMPT_TOKENS:]
            prompt_token_ids_list.append(ids)

        vllm_inputs = [{"prompt_token_ids": ids} for ids in prompt_token_ids_list]
        for input_data in vllm_inputs:
            if not isinstance(input_data["prompt_token_ids"], list):
                input_data["prompt_token_ids"] = list(input_data["prompt_token_ids"])

        # 与 vllm_rollout_spmd 一致：kwargs 构建 + val_kwargs 覆盖；不传 seed 以与 client API 行为一致（非确定性采样）
        kwargs = dict(
            n=1,
            logprobs=0,
            max_tokens=config.response_length,
        )
        kwargs["detokenize"] = False
        for k in config.keys():
            if k in ("val_kwargs", "seed"):
                continue
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)
        kwargs["top_k"] = config.val_kwargs.top_k
        kwargs["top_p"] = config.val_kwargs.top_p
        kwargs["temperature"] = config.val_kwargs.temperature
        sampling_params = SamplingParams(**kwargs)

        outputs = llm.generate(
            prompts=vllm_inputs,
            sampling_params=sampling_params,
            use_tqdm=False,
        )

        # 与 vllm_rollout_spmd 一致：response 收集方式
        response = []
        for output in outputs:
            for sample_id in range(len(output.outputs)):
                response_ids = output.outputs[sample_id].token_ids
                response.append(response_ids)

        # 与 vllm_rollout_spmd 一致：pad_2d_list_to_length
        response_padded = pad_2d_list_to_length(response, pad_id, max_length=config.response_length)

        # 从 padded 取每行，截断到首个 EOS/PAD 再解码（与 tokenized 一致用同一 tokenizer decode）
        for i in range(len(active_indices)):
            original_idx = active_indices[i]
            row = response_padded[i].tolist()
            content_ids = []
            for tid in row:
                if tid == eos_token_id or tid == pad_id:
                    break
                content_ids.append(tid)
            text = enc_dec_tok.decode(content_ids, skip_special_tokens=True)
            actions[original_idx] = text.strip() if text else ""

    except Exception as e:
        print(f"❌ vLLM generate failed: {e}")
        for idx in active_indices:
            if actions[idx] is None:
                actions[idx] = ""

    for i in range(total):
        if actions[i] is None:
            actions[i] = ""

    return actions


def _truncate_obs_for_save(obs: str, max_len: Optional[int]) -> str:
    """写入结果时对单条 observation 做长度截断。"""
    if obs is None or max_len is None or max_len <= 0:
        return obs if obs is not None else ""
    if len(obs) <= max_len:
        return obs
    return f"[truncated, total {len(obs)} chars] ..." + obs[-max_len:]


def main():
    args = SpecificArgs()

    if not os.path.exists(args.input_query_dir):
        print(f"❌ Input file not found: {args.input_query_dir}")
        return

    llm, tokenizer = get_vllm_llm_and_tokenizer()
    if llm is None or tokenizer is None:
        print("❌ 必须成功加载 vLLM 与 Tokenizer，程序退出。")
        return
    hf_tokenizer = get_hf_tokenizer()
    if hf_tokenizer is None:
        print("⚠️ HF Tokenizer 未加载，将用 vLLM tokenizer 做 prompt/decode，与 client 端可能不一致")

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

    print("🚀 初始化 ToolBench 全量评测 (vLLM 进程内 generate + 本地 Tokenize，与 serve tokenized 对比)")
    print(f"Dataset Size: {total_dataset_len} | Batch Size: {ENV_NUM} | Group N: {GROUP_N}")

    group_stats = [[] for _ in range(GROUP_N)]
    global_processed_count = 0
    global_final_rewards = []
    # 收集每个 episode 的 observation 轨迹（仅当 COLLECT_OBSERVATIONS 为 True 时填充）
    global_episode_observations = []

    pbar_total = min(total_dataset_len, MAX_SAMPLES) if MAX_SAMPLES is not None else total_dataset_len
    pbar = tqdm(total=pbar_total, desc="Total Progress", unit="sample")

    while True:
        if global_processed_count >= MAX_SAMPLES:
            print(f"✅ 已达到指定样本数量 {MAX_SAMPLES}，停止测试。")
            break
        # import pdb; pdb.set_trace()
        observations, infos = envs.reset()

        current_batch_size = len(observations)
        batch_group_ids = []
        for i in range(current_batch_size):
            abs_index = global_processed_count + i
            group_id = abs_index % GROUP_N
            batch_group_ids.append(group_id)

        global_processed_count += current_batch_size
        dones = [False] * current_batch_size
        episode_rewards = [0.0] * current_batch_size
        final_rewards = [None] * current_batch_size
        # 本 batch 内每个 env 的 observation 轨迹（仅当 COLLECT_OBSERVATIONS 时使用）
        episode_obs_trajectory = [[] for _ in range(current_batch_size)]
        # 本 batch 的轨迹按 env 索引暂存，最后按序 extend，保证与 global_final_rewards 索引一致
        batch_episode_observations = [None] * current_batch_size if COLLECT_OBSERVATIONS else None

        if COLLECT_OBSERVATIONS:
            for i in range(current_batch_size):
                episode_obs_trajectory[i].append(observations[i])

        step_cnt = 0
        max_steps = 15  # eval.sh: env.max_steps=15

        while not all(dones) and step_cnt < max_steps:
            step_cnt += 1
            actions = generate_actions_vllm_direct(llm, tokenizer, hf_tokenizer, observations, dones, args)
            next_obs, step_rewards, next_dones, step_infos = envs.step(actions)

            for i in range(current_batch_size):
                if not dones[i]:
                    episode_rewards[i] += step_rewards[i]
                    if COLLECT_OBSERVATIONS:
                        episode_obs_trajectory[i].append(next_obs[i])
                    if next_dones[i]:
                        final_rewards[i] = episode_rewards[i]
                        dones[i] = True
                        if COLLECT_OBSERVATIONS:
                            trajectory = [_truncate_obs_for_save(o, MAX_OBS_LENGTH_SAVED) for o in episode_obs_trajectory[i]]
                            batch_episode_observations[i] = trajectory
                    observations[i] = next_obs[i]

        # 未在循环内结束的 episode（如达到 max_steps）也写入本 batch 的对应槽位
        if COLLECT_OBSERVATIONS:
            for i in range(current_batch_size):
                if batch_episode_observations[i] is not None:
                    continue
                if not episode_obs_trajectory[i]:
                    continue
                trajectory = [_truncate_obs_for_save(o, MAX_OBS_LENGTH_SAVED) for o in episode_obs_trajectory[i]]
                batch_episode_observations[i] = trajectory
            global_episode_observations.extend(batch_episode_observations)

        import pdb

        pdb.set_trace()

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
    print("📊 Final Evaluation Report (vLLM Direct Generate + Tokenized)")
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

    result_file = f"eval_result_grouped_vllm_direct_{int(time.time())}.json"
    result_data = {
        "total_evaluated": total_evaluated,
        "overall_average": overall_avg,
        "group_n": GROUP_N,
        "group_details": group_results_json,
        "all_rewards_flat": global_final_rewards,
        "backend": "vllm_direct_generate_tokenized",
    }
    if COLLECT_OBSERVATIONS and global_episode_observations:
        result_data["episode_observations"] = global_episode_observations
        result_data["max_obs_length_saved"] = MAX_OBS_LENGTH_SAVED
    with open(result_file, "w") as f:
        json.dump(result_data, f, indent=4)
    print(f"\n📝 详细分组结果已保存至: {result_file}")
    if COLLECT_OBSERVATIONS and global_episode_observations:
        print(f"   (已收集 {len(global_episode_observations)} 条 episode 的 observation 轨迹)")


if __name__ == "__main__":
    # 避免 "Cannot re-initialize CUDA in forked subprocess"：强制使用 spawn，
    # 使后续 vLLM/多进程使用 spawn 而非 fork，与 CUDA 兼容。
    import multiprocessing

    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # 已设置过则忽略
    main()
