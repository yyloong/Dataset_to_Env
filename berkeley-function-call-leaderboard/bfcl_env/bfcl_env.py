"""
BFCL Environment for Reinforcement Learning

该模块将BFCL (Berkeley Function Calling Leaderboard) 封装为RL环境,
复用BFCL原有的评测逻辑,支持所有BFCL任务类别。
"""

import os
import random
import shutil
from collections import defaultdict

from bfcl_eval._llm_response_generation import build_handler, get_involved_test_entries
from bfcl_eval.constants.category_mapping import MEMORY_SCENARIO_NAME
from bfcl_eval.constants.eval_config import RESULT_PATH
from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING
from bfcl_eval.eval_checker.eval_runner_helper import load_ground_truth_entry
from bfcl_eval.utils import (
    extract_test_category_from_id,
    is_format_sensitivity,
    is_memory,
    is_memory_prereq,
    load_dataset_entry,
    populate_initial_settings_for_memory_test_cases,
    populate_initial_settings_for_web_search_test_cases,
    sort_key,
)
from memory_env import BFCLMemoryEnv
from single_bfcl_env import BFCLEnv

default_memory_data_dir = os.path.join(os.path.dirname(__file__), "memory_data")


# ========== Memory env grouping helpers (for BFCLMemoryEnv) ==========
def _extract_memory_scenario_from_entry(entry: dict) -> str | None:
    """
    从 memory entry 中提取 scenario (如 customer, finance)。
    prereq: id 格式为 memory_kv_prereq_0-customer-0，scenario 在 '-' 之后
    test: 使用 entry["scenario"]
    """
    entry_id = entry.get("id", "")
    if is_memory_prereq(entry_id) and "-" in entry_id:
        parts = entry_id.split("-")
        if len(parts) >= 2:
            return parts[1].split("-")[0] if "-" in parts[1] else parts[1]
    return entry.get("scenario")


def _group_memory_entries_by_scenario(memory_entries: list[dict], test_category: str) -> list[dict]:
    """
    将 memory entries 按 (test_category, scenario) 分组，
    每组内按依赖关系排序: prereq_0..prereq_n, test_0..test_m。
    """
    groups = defaultdict(list)
    for entry in memory_entries:
        scenario = _extract_memory_scenario_from_entry(entry)
        if scenario and scenario in MEMORY_SCENARIO_NAME:
            key = (test_category, scenario)
            groups[key].append(entry)

    result = []
    for (cat, scenario), entries in groups.items():
        sorted_entries = sorted(entries, key=sort_key)
        result.append(
            {
                "entries": sorted_entries,
                "scenario": scenario,
                "memory_test_category": cat,
                "id": f"{cat}_{scenario}_env",
            }
        )
    return result


def build_bfclv4_envs(
    env_name,
    seed,
    env_num,
    group_n,
    resources_per_worker,
    model_name,
    is_train=True,
):
    """
    构建BFCL环境

    Args:
        env_name: 环境名称,格式为'bfcl-{test_category}'
        seed: 随机种子
        env_num: 环境数量
        group_n: 每个环境的样本数量
        resources_per_worker: 资源配置
        model_name: 模型名称
        is_train: 是否为训练模式

    Returns:
        环境列表
    """
    test_category = env_name.split("-")[1:]
    (
        all_test_categories,
        all_test_entries_involved,
    ) = get_involved_test_entries(test_category, None)

    # memory 与其它类别不能同时使用
    memory_cats = [c for c in all_test_categories if is_memory(c)]
    non_memory_cats = [c for c in all_test_categories if not is_memory(c)]
    if memory_cats and non_memory_cats:
        raise ValueError(
            "memory 类别不能与其它 test category 同时使用。"
            f"当前包含 memory 类别: {memory_cats}，非 memory 类别: {non_memory_cats}。"
            "请分别单独指定 memory 或其它类别。"
        )

    all_eval_entries = []
    for test_category in all_test_categories:
        all_eval_entries.extend(load_dataset_entry(test_category, include_prereq=True, include_language_specific_hint=False))

    if model_name not in MODEL_CONFIG_MAPPING:
            raise ValueError(f"Unknown model_name '{model_name}'.\n• For officially supported models, please refer to `SUPPORTED_MODELS.md`.\n• For running new models, please refer to `README.md` and `CONTRIBUTING.md`.")
    print(f"Building BFCL environments for {model_name}")

    if any(is_format_sensitivity(test_category) for test_category in all_test_categories):
        if MODEL_CONFIG_MAPPING[model_name].is_fc_model:
            print(f"⚠️ Warning: Format sensitivity test cases are only supported for prompting (non-FC) models. Since {model_name} is a FC model based on its config, the format sensitivity test cases will be skipped.")

    if any(is_format_sensitivity(test_category) for test_category in all_test_categories) and MODEL_CONFIG_MAPPING[model_name].is_fc_model:
        test_cases_to_generate = [test_case for test_case in all_test_entries_involved if not is_format_sensitivity(test_case["id"])]
    else:
        test_cases_to_generate = all_test_entries_involved

    test_cases_to_generate = populate_initial_settings_for_web_search_test_cases(test_cases_to_generate)

    # ========== memory 类别: 按 scenario 分组，使用 BFCLMemoryEnv ==========
    use_memory_env = any(is_memory(cat) for cat in all_test_categories)
    if use_memory_env:
        # test_cases_to_generate = clean_up_memory_prereq_entries(test_cases_to_generate)
        model_name_dir = model_name.replace("/", "_")
        model_result_dir = RESULT_PATH / model_name_dir
        memory_result_path = model_result_dir / "agentic" / "memory"
        if memory_result_path.exists():
            shutil.rmtree(memory_result_path)
        test_cases_to_generate = populate_initial_settings_for_memory_test_cases(test_cases_to_generate, model_result_dir)
        prompt_entries_total = []
        for cat in all_test_categories:
            if is_memory(cat):
                cat_entries = [e for e in test_cases_to_generate if extract_test_category_from_id(e.get("id", ""), remove_prereq=True) == cat.replace("_prereq", "")]
                groups = _group_memory_entries_by_scenario(cat_entries, cat)
                prompt_entries_total.extend(groups)
        all_eval_entries = prompt_entries_total
        if env_num == 0:
            env_num = len(prompt_entries_total)
            print(f"env_num is 0, set env_num to memory groups: {env_num}")
    else:
        prompt_entries_total = test_cases_to_generate
        if env_num == 0:
            env_num = len(test_cases_to_generate)
            print(f"env_num is 0, set env_num to the size of dataset: {env_num}")
    # ========== end memory ==========

    all_test_categories_final = all_test_categories
    model_name_final = model_name
    # 加载ground truth数据
    ground_truth_dict = {}
    for category in all_test_categories:
        try:
            ground_truth_entries = load_ground_truth_entry(category)
            for entry in ground_truth_entries:
                if entry["id"].rsplit("_", 1)[0] == "web_search":
                    id = entry["id"].rsplit("_", 1)[1]
                    ground_truth_dict[category + "_" + id] = entry
                # ========== memory: ground truth 原 id 为 memory_0-customer-0，test entry 为 memory_kv_0 等 ==========
                elif is_memory(category) and entry["id"].startswith("memory_"):
                    backend_id = entry["id"].replace("memory_", f"{category}_", 1)
                    ground_truth_dict[backend_id] = entry
                else:
                    ground_truth_dict[entry["id"]] = entry
                # ========== end memory ==========
        except Exception as e:
            print(f"Warning: Failed to load ground truth for category {category}: {e}")

    # 构建handler用于评测
    handler = build_handler(model_name_final, temperature=0.0)

    random.seed(a=seed, version=2)

    print(f"len(prompt_entries_total): {len(prompt_entries_total)}")
    if not is_train and len(prompt_entries_total) % env_num != 0:
        # 用黄色打印警告
        print("\033[93mWarning: In evaluation mode, the number of prompt entries is not divisible by the number of envs. The last batch will be truncated.\033[0m")
        print("\033[93mIf you want to use full evaluation data, please set the number of envs to be a multiple of the number of prompt entries.\033[0m")
        prompt_entries_total = prompt_entries_total[: len(prompt_entries_total) - len(prompt_entries_total) % env_num]
        all_eval_entries = all_eval_entries[: len(all_eval_entries) - len(all_eval_entries) % env_num]
        print(f"Truncated prompt entries to: {len(prompt_entries_total)}")

    # ========== memory 类别使用 BFCLMemoryEnv ==========
    if use_memory_env:
        assert group_n == 1, "memory env only supports group_n = 1"
        envs = [
            BFCLMemoryEnv(
                env_id=i,
                model_name=model_name_final,
                handler=handler,
                ground_truth_dict=ground_truth_dict,
            )
            for i in range(env_num)
        ]
    else:
        envs = [
            BFCLEnv(
                group_n=group_n,
                env_id=i,
                model_name=model_name_final,
                handler=handler,
                ground_truth_dict=ground_truth_dict,
            )
            for i in range(env_num)
        ]
    # ========== end memory ==========
    return BFCLVectorEnv(
        envs=envs,
        resources_per_worker=resources_per_worker,
        by_turns=not is_train,
        eval_prompt_entries_total=all_eval_entries,
        prompt_entries_total=prompt_entries_total,
        all_test_categories=all_test_categories_final,
    )


class BFCLVectorEnv:
    """
    向量化的BFCL环境包装器 - 纯串行执行版本(评估时间很短,不需要并行)
    如果并行处理需要先对execute_multi_turn_func_call进行重写，避免不同idx之间的状态相互影响
    """

    def __init__(
        self,
        envs,
        resources_per_worker=None,
        by_turns=False,
        prompt_entries_total=None,
        eval_prompt_entries_total=None,
        all_test_categories=None,
    ):
        self.envs = envs
        self.num_envs = len(envs)
        self.by_turns = by_turns
        self.resources_per_worker = resources_per_worker
        self.now_index = 0
        self.prompt_entries_total = prompt_entries_total or []
        self.all_test_categories = all_test_categories or []
        self.eval_prompt_entries_total = eval_prompt_entries_total or []
        assert len(self.prompt_entries_total) == len(self.eval_prompt_entries_total), "prompt_entries_total and eval_prompt_entries_total must have the same length"
        self.entries_combine = [(self.prompt_entries_total[i], self.eval_prompt_entries_total[i]) for i in range(len(self.prompt_entries_total))]
        print("使用串行模式执行")

    def select_random(self):
        return random.sample(self.entries_combine, self.num_envs)

    def select_sequential(self):
        if self.now_index >= len(self.entries_combine):
            self.now_index = 0
            raise StopIteration
        entries = self.entries_combine[self.now_index : self.now_index + self.num_envs]
        self.now_index += self.num_envs
        return entries

    def reset(self):
        if self.by_turns:
            entries = self.select_sequential()
        else:
            entries = self.select_random()
        observations = []
        infos = []
        for i, env in enumerate(self.envs):
            obs, info = env.reset(entries[i])
            observations.extend(obs)
            infos.extend(info)
        return observations, infos

    def step(self, actions):
        """
        执行动作（串行执行）
        """
        if self.num_envs == 0:
            return [], [], [], []

        group_n = getattr(self.envs[0], "group_n", None)
        if group_n is None:
            raise ValueError("BFCLVectorEnv.step: env 缺少 group_n 属性")

        if not isinstance(actions, (list, tuple)):
            raise ValueError("BFCLVectorEnv.step: actions 必须是 list/tuple")
        expected = self.num_envs * group_n
        if len(actions) != expected:
            raise ValueError(f"BFCLVectorEnv.step: actions 长度为 {len(actions)}，期望 {expected}")

        # 输出容器
        observations = [None] * expected
        rewards = [0.0] * expected
        dones = [False] * expected
        infos = [{}] * expected

        print("串行执行模式")
        print(f"Total environments: {self.num_envs}, group_n: {group_n}")

        # 串行执行每个环境的所有动作
        for env_i in range(self.num_envs):
            env = self.envs[env_i]

            # 对当前环境的每个entry执行动作
            for idx in range(group_n):
                flat_i = env_i * group_n + idx
                act = actions[flat_i]

                try:
                    obs, rew, done, info = env.step_single_entry(idx, act)
                    observations[flat_i] = obs
                    rewards[flat_i] = float(rew) if rew is not None else 0.0
                    dones[flat_i] = bool(done)
                    infos[flat_i] = info if isinstance(info, dict) else {"info": info}
                except Exception as e:
                    import traceback

                    tb = traceback.format_exc()
                    print(f"Environment {env_i}, entry {idx} failed: {e}\n{tb}")
                    observations[flat_i] = "Error"
                    rewards[flat_i] = 0.0
                    dones[flat_i] = True
                    infos[flat_i] = {"error": str(e), "error_trace": tb}

        return observations, rewards, dones, infos
