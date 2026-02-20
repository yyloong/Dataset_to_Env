import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

# 尝试导入环境
from bfcl_env import build_bfclv4_envs
from openai import OpenAI
from tqdm import tqdm

# ================= 配置区域 =================
# 1. BFCL 环境配置
BFCL_MODEL_KEY = "Qwen/Qwen3-8B"
#TEST_CATEGORY = "simple_java-simple_python-simple_javascript-parallel-multiple-parallel_multiple"
#TEST_CATEGORY = "live_simple-live_multiple-live_parallel-live_parallel_multiple"
TEST_CATEGORY = "live_irrelevance-live_relevance-irrelevance"
#TEST_CATEGORY = "memory"
#TEST_CATEGORY = "web_search"
#TEST_CATEGORY = "format_sensitivity"
#TEST_CATEGORY = "simple_java"
#TEST_CATEGORY = "multi_turn"

ENV_NUM = 15 # 每个 batch 处理的样本数（与 bfcl_env_test 保持一致语义）
GROUP_N = 1
SEED = 42
MAX_SAMPLES: Optional[int] = None # 最多评估多少个样本，None 表示跑完整个数据集

# 2. 本地 vLLM 配置
os.environ["NO_PROXY"] = "localhost,127.0.0.1"
VLLM_API_URL = "http://127.0.0.1:8000/v1"
VLLM_MODEL_NAME = "Qwen/Qwen3-8B"
MODEL_PATH = "Qwen/Qwen3-8B"

# 与 BFCL 原生评测一致：使用 stop 避免模型在 <|im_end|> 后继续生成，导致解析失败
STOP_TOKENS = ["<|im_end|>", "<|endoftext|>", "</s>"]

# 多线程配置（仍然是“单条 prompt 调一次 API”，只是并行很多条）
MAX_WORKERS = 100  # 最大线程数
# ===========================================


def get_vllm_client():
    return OpenAI(base_url=VLLM_API_URL, api_key="EMPTY")


def generate_single_action(
    client: OpenAI,
    prompt: str,
    index: int,
) -> tuple:
    """
    单个请求函数，用于多线程调用（保持“单条调用 API”的形式）

    Args:
        client: OpenAI 客户端
        prompt: 字符串格式的提示词
        index: 原始索引

    Returns:
        (index, action) 元组
    """
    try:
        response = client.completions.create(
            model=VLLM_MODEL_NAME,
            prompt=prompt,  # 直接使用字符串 prompt
            max_tokens=4096,
            temperature=0.001,  # 与 BFCL 原生一致；temperature=1 会导致结构化输出几乎全错、simple_java 准确率为 0
            stop=STOP_TOKENS,
        )
        action = response.choices[0].text.strip()
        return (index, action)
    except Exception as e:
        tqdm.write(f"⚠️ 请求失败 (index {index}): {e}")
        return (index, "")


def generate_actions_openai_batch(
    client: OpenAI,
    observations: List[str],
    dones: List[bool],
) -> List[str]:
    """
    使用多线程方式并发请求 vLLM API
    每个请求发送字符串格式的 prompt（而非 token IDs）
    注意：这里的“batch”只是线程并发，仍然是对每条 prompt 单独调用一次 API。

    Args:
        client: OpenAI 客户端
        observations: 观察列表
        dones: 完成状态列表

    Returns:
        动作列表
    """
    total = len(observations)
    actions = [None] * total

    # 收集需要处理的请求任务
    tasks = []
    for i, (obs, done) in enumerate(zip(observations, dones)):
        if done:
            actions[i] = "FINISHED"
        elif not obs:
            actions[i] = ""
        else:
            tasks.append((i, obs))

    if not tasks:
        # 填充剩余的 None 为空字符串
        for i in range(total):
            if actions[i] is None:
                actions[i] = ""
        return actions

    # 使用多线程并行请求
    max_workers = min(len(tasks), MAX_WORKERS)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_index = {executor.submit(generate_single_action, client, obs, idx): idx for idx, obs in tasks}

        # 收集结果
        for future in as_completed(future_to_index):
            try:
                idx, action = future.result()
                actions[idx] = action
            except Exception as e:
                original_idx = future_to_index[future]
                tqdm.write(f"⚠️ 线程执行失败 (index {original_idx}): {e}")
                actions[original_idx] = ""

    # 确保所有位置都有值
    for i in range(total):
        if actions[i] is None:
            actions[i] = ""

    return actions


def main():
    print(f"🚀 初始化 BFCL 评测 | Model: {BFCL_MODEL_KEY} | Batch EnvNum: {ENV_NUM}")

    client = get_vllm_client()

    try:
        # 这里的 env_num 语义与 bfcl_env_test.py 保持一致：单次 reset 产生一个 batch
        envs = build_bfclv4_envs(
            env_name=f"bfcl-{TEST_CATEGORY}",
            seed=SEED,
            env_num=ENV_NUM,
            group_n=GROUP_N,
            resources_per_worker={},
            model_name=BFCL_MODEL_KEY,
            is_train=False,
        )
    except Exception as e:
        print(f"❌ 环境构建失败: {e}")
        return

    # 获取数据集总大小（与 bfcl_env_test.py 对齐）
    total_dataset_len = len(envs.prompt_entries_total) if hasattr(envs, "prompt_entries_total") else None
    if total_dataset_len is None:
        print("⚠️ 无法获取数据集总大小，将使用当前 batch 大小")
        total_dataset_len = ENV_NUM if ENV_NUM > 0 else 1

    print(f"📊 Dataset Size: {total_dataset_len} | Batch Size: {ENV_NUM} | Group N: {GROUP_N}")

    # ===== 全局统计（跨 batch）=====
    global_processed_count = 0
    # 按 test_category 分组统计 reward
    category_stats = {}  # {test_category: {"rewards": [], "count": 0}}

    pbar_total = min(total_dataset_len, MAX_SAMPLES) if MAX_SAMPLES is not None else total_dataset_len
    pbar = tqdm(total=pbar_total, desc="Total Progress", unit="sample", ncols=100)

    max_steps = 300

    # ================= 外层循环：处理多个 batch =================
    while True:
        if MAX_SAMPLES is not None and global_processed_count >= MAX_SAMPLES:
            print(f"\n✅ 已达到指定样本数量 {MAX_SAMPLES}，停止测试。")
            break

        try:
            print("🔄 Environment Reset...")
            observations, infos = envs.reset()
        except StopIteration:
            print("\n✅ 已处理完所有数据，停止测试。")
            break
        except Exception as e:
            print(f"\n⚠️ Environment Reset 出错: {e}，停止测试。")
            break

        current_batch_size = len(observations)

        if current_batch_size == 0:
            print("\n✅ 没有更多数据，停止测试。")
            break

        # 如果当前 batch 会超过 MAX_SAMPLES，则只处理部分样本
        if MAX_SAMPLES is not None:
            remaining = MAX_SAMPLES - global_processed_count
            if remaining <= 0:
                break
            if current_batch_size > remaining:
                observations = observations[:remaining]
                current_batch_size = remaining

        # 收集当前 batch 每个样本的 test_category
        batch_categories = []
        for info in infos:
            test_category = info.get("test_category")
            batch_categories.append(test_category)

        dones = [False] * current_batch_size
        episode_rewards = [0.0] * current_batch_size
        final_rewards = [None] * current_batch_size
        step_cnt = 0

        print(f"📊 当前 batch 大小: {current_batch_size}，最大步数 {max_steps}...")

        # ================= 内层循环：处理单个 batch 的 episode =================
        while not all(dones) and step_cnt < max_steps:
            step_cnt += 1

            # 计算当前未完成的数量（基于按 instance 下标的 final_rewards）
            n_done_so_far = sum(1 for r in final_rewards if r is not None)
            active_count = current_batch_size - n_done_so_far
            tqdm.write(f"Batch Step {step_cnt}/{max_steps} | Active Envs: {active_count} | Generating Actions...")

            # === 生成动作（保持“单条调用 API”的多线程版本） ===
            actions = generate_actions_openai_batch(
                client,
                observations,
                dones,
            )

            # === 环境步进 ===
            next_obs, step_rewards, next_dones, step_infos = envs.step(actions)

            # 统计本轮有多少个变成 Done
            finished_in_this_step = 0

            # === 数据更新 ===
            for i in range(current_batch_size):
                if not dones[i]:
                    episode_rewards[i] += step_rewards[i]

                    if next_dones[i]:
                        final_rewards[i] = episode_rewards[i]
                        finished_in_this_step += 1

                    observations[i] = next_obs[i]
                    dones[i] = next_dones[i]

        # ================= 收集当前 batch 的结果 =================
        for i in range(current_batch_size):
            reward = final_rewards[i] if final_rewards[i] is not None else 0.0

            # 按 test_category 统计
            test_category = batch_categories[i]
            if test_category not in category_stats:
                category_stats[test_category] = {"rewards": [], "count": 0}
            category_stats[test_category]["rewards"].append(reward)
            category_stats[test_category]["count"] += 1

        global_processed_count += current_batch_size

        # 更新进度条
        pbar.update(current_batch_size)
        # 计算所有 category 的总体平均用于进度条显示
        total_rewards = []
        for stats in category_stats.values():
            total_rewards.extend(stats["rewards"])
        curr_avg = sum(total_rewards) / len(total_rewards) if total_rewards else 0.0
        pbar.set_postfix({"Avg": f"{curr_avg:.3f}", "Processed": f"{global_processed_count}"})

    pbar.close()

    # =================================================

    print("\n" + "=" * 70)
    print("📊 Final Evaluation Report (by test_category)")
    print("=" * 70)

    # 计算总体统计
    total_rewards = []
    for stats in category_stats.values():
        total_rewards.extend(stats["rewards"])
    
    total_evaluated = len(total_rewards)
    finished_count = sum(1 for r in total_rewards if r > 0.0)
    overall_pass_rate = finished_count / total_evaluated if total_evaluated > 0 else 0.0

    print(f"Total Instances Processed: {total_evaluated}")
    print(f"Overall Pass Rate        : {overall_pass_rate:.2%} ({finished_count}/{total_evaluated})")

    # 按 test_category 打印统计
    if category_stats:
        print("-" * 70)
        print(f"{'Category':<30} | {'Count':<8} | {'Avg Reward':<12} | {'Pass Rate':<12}")
        print("-" * 70)

        # 按 category 名称排序
        for category in sorted(category_stats.keys()):
            stats = category_stats[category]
            rewards = stats["rewards"]
            count = len(rewards)
            if count > 0:
                avg_r = sum(rewards) / count
                pass_r = sum(1 for r in rewards if r >= 1.0) / count
            else:
                avg_r = 0.0
                pass_r = 0.0

            # 截断过长的 category 名称
            display_category = category[:28] + ".." if len(category) > 30 else category
            print(f"{display_category:<30} | {count:<8} | {avg_r:.4f}       | {pass_r:.2%}")

    print("=" * 70)


if __name__ == "__main__":
    main()
