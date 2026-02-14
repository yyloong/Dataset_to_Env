import json
import os

from bfcl_env import build_bfclv4_envs
from tqdm import tqdm

# ================= 配置区域 =================
# 1. BFCL 环境配置
BFCL_MODEL_KEY = "Qwen/Qwen3-8B"
TEST_CATEGORY = "web_search_base"
ENV_NUM = 0  # 并行环境数
GROUP_N = 1
SEED = 42
RESULT_FILE = os.path.join(os.path.dirname(__file__), "result", BFCL_MODEL_KEY.replace("/", "_"), "agentic", f"BFCL_v4_{TEST_CATEGORY}_result.json")


def construct_actions(result_file_path):
    actions = []
    with open(result_file_path) as f:
        for i, line in enumerate(f):
            result = json.loads(line)
            if isinstance(result["result"][0], list):
                actions.append([result["result"][i][j] for i in range(len(result["result"])) for j in range(len(result["result"][i]))])
            else:
                actions.append([result["result"]])
        # 补齐到相同长度
        print(f"length of actions: {len(actions)}")
        max_length = max(len(action) for action in actions)
        # 补齐到相同长度
        for action in actions:
            if len(action) < max_length:
                action.extend(["FINISHED"] * (max_length - len(action)))
        # repeat actions for more than one group
        new_actions = []
        for i in range(len(actions)):
            new_actions.extend([actions[i]] * GROUP_N)

        return new_actions


def main():
    print(f"🚀 初始化 BFCL 评测 | Model: {BFCL_MODEL_KEY} | EnvNum: {ENV_NUM}")

    try:
        envs = build_bfclv4_envs(
            env_name=f"bfcl-{TEST_CATEGORY}",
            seed=SEED,
            env_num=ENV_NUM,
            group_n=GROUP_N,
            resources_per_worker={"num_cpus": 10},
            model_name=BFCL_MODEL_KEY,
            is_train=False,
        )
    except Exception as e:
        print(f"❌ 环境构建失败: {e}")
        return

    print("🔄 Environment Reset...")
    observations, infos = envs.reset()

    total_instances = len(observations)
    dones = [False] * total_instances

    episode_rewards = [0.0] * total_instances
    # 按 instance 下标存储最终 reward，用于正确分组（避免按完成顺序分组）
    final_rewards = [None] * total_instances

    step_cnt = 0
    max_steps = 30

    # ================= 修改后的主循环 =================
    # 调整 ncols 让进度条在不同终端下显示更舒服
    print(f"📊 开始评测，共 {total_instances} 个实例，最大步数 {max_steps}...")
    actions = construct_actions(RESULT_FILE)

    with tqdm(total=total_instances, desc="Evaluated", unit="ep", ncols=100) as pbar:
        while not all(dones) and step_cnt < max_steps:
            step_cnt += 1

            # 1. 打印步数信息 (关键：使用 tqdm.write)
            # 计算当前未完成的数量（基于按 instance 下标的 final_rewards）
            n_done_so_far = sum(1 for r in final_rewards if r is not None)
            active_count = total_instances - n_done_so_far
            tqdm.write(f"Step {step_cnt}/{max_steps} | Active Envs: {active_count} | Generating Actions...")

            now_actions = [actions[i][step_cnt - 1] for i in range(total_instances)]

            # === 环境步进 ===
            next_obs, step_rewards, next_dones, step_infos = envs.step(now_actions)
            # 计算有多少给step_infos是空的
            empty_infos_count = sum(1 for info in step_infos if info == {})
            print(f"Empty infos count: {empty_infos_count}")
            # 统计本轮有多少个变成 Done
            finished_in_this_step = 0

            # === 数据更新 ===
            for i in range(total_instances):
                # 只处理之前未完成的
                if not dones[i]:
                    episode_rewards[i] += step_rewards[i]

                    # 如果这一步刚变成 Done：按 instance 下标写入 final_rewards
                    if next_dones[i]:
                        final_rewards[i] = episode_rewards[i]
                        finished_in_this_step += 1

                    # 更新状态
                    observations[i] = next_obs[i]
                    dones[i] = next_dones[i]

            # 2. 统一更新进度条与 Done：基于 final_rewards（按 instance 下标）

            if finished_in_this_step > 0:
                n_done = sum(1 for r in final_rewards if r is not None)
                pbar.n = n_done
                pbar.refresh()
                done_rewards = [r for r in final_rewards if r is not None]
                current_avg = sum(done_rewards) / n_done if n_done > 0 else 0.0
                pbar.set_postfix(
                    {
                        "AvgRw": f"{current_avg:.2f}",
                        "Done": f"{n_done}/{total_instances}",
                    }
                )

    # =================================================

    print("\n" + "=" * 40)
    print("📊 Evaluation Summary")
    print("=" * 40)
    for i in range(GROUP_N):
        # 按 instance 下标分组：flat i % GROUP_N == group_idx
        single_group_rewards = [final_rewards[j] for j in range(i, total_instances, GROUP_N) if final_rewards[j] is not None]
        group_avg = sum(single_group_rewards) / len(single_group_rewards) if single_group_rewards else 0.0
        print(f"Group {i} Avg Reward: {group_avg:.6f}, Group Size: {len(single_group_rewards)}")

    finished_count = sum(1 for r in final_rewards if r is not None)
    done_rewards = [r for r in final_rewards if r is not None]
    avg_reward = sum(done_rewards) / finished_count if finished_count > 0 else 0.0

    print(f"Total Steps      : {step_cnt}")
    print(f"Total Instances  : {total_instances}")
    print(f"Finished         : {finished_count}")
    print(f"Average Reward   : {avg_reward:.6f}")

    # 对比group 内的reward是否相同
    if GROUP_N > 1:
        for i in range(total_instances // GROUP_N):
            group_rewards = [final_rewards[j] for j in range(i * GROUP_N, (i + 1) * GROUP_N) if final_rewards[j] is not None]
            if len(group_rewards) > 0:
                if group_rewards[0] != group_rewards[1]:
                    print(f"Group {i} rewards are not the same")
                    print(group_rewards)

    if finished_count < total_instances:
        print(f"⚠️ {total_instances - finished_count} instances did not finish.")


if __name__ == "__main__":
    main()
