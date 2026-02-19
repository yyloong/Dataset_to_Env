import re
from copy import deepcopy

from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler
from bfcl_eval.utils import (
    contain_multi_turn_interaction,
    extract_test_category_from_id,
    is_memory_prereq,
)

from single_bfcl_env import BFCLEnv


# ========== BFCLMemoryEnv: memory 类别专用，同一 scenario 的 prereq+test 串行执行 ==========
class BFCLMemoryEnv(BFCLEnv):
    """
    继承 BFCLEnv，针对 memory 类别的特殊性：
    - 同一 scenario (如 customer) 的 prereq 和 test 视为一个 env
    - prereq 结束后: score=0.0, done=False, 切换到对应 test case
    - test case 结束后: 评估打分但 done=False，直到该类别所有 test 完成才 done=True
    """

    def __init__(
        self,
        env_id,
        model_name=None,
        handler=None,
        ground_truth_dict=None,
    ):
        super().__init__(
            group_n=1,
            env_id=env_id,
            model_name=model_name,
            handler=handler,
            ground_truth_dict=ground_truth_dict,
        )

    def reset(self, entry):
        """entry 为 group 格式: {"entries": [...], "scenario": str, "memory_test_category": str}"""
        # memory env eval entry and test entry are the same
        entry = entry[0]
        self.memory_group_entries = entry["entries"]
        self.current_memory_entry_idx = 0
        self.memory_test_scores = []
        sub_entry = self.memory_group_entries[0]
        return self._reset_for_sub_entry(sub_entry)

    def _reset_for_sub_entry(self, sub_entry):
        """为单个 sub_entry 设置内部状态并返回初始 observation"""
        self.entries = [deepcopy(sub_entry) for _ in range(self.group_n)]
        # memory env eval entry and test entry are the same
        self.eval_entries = self.entries
        self.entry_id = sub_entry["id"]
        self.entry_category = extract_test_category_from_id(self.entry_id)
        self.ground_truth = self.ground_truth_dict.get(self.entry_id, {})
        self.is_multi_turn_interaction = contain_multi_turn_interaction(self.entry_id)
        self.is_FC = self.handler.is_fc_model if self.handler else False
        self.is_OSS = isinstance(self.handler, OSSHandler) if self.handler else False
        self.initial_config = sub_entry.get("initial_config", {})
        self.involved_classes = sub_entry["involved_classes"]
        self.holdout_function = sub_entry.get("missed_function", {})
        self.all_multi_turn_messages = sub_entry["question"]
        self.all_model_response = [[[] for _ in range(len(self.all_multi_turn_messages))]]
        self.all_reasoning_content = [[[] for _ in range(len(self.all_multi_turn_messages))]]
        self.turns = [0]
        self.steps = [0]
        self.dones = [False]
        self.inference_data = [{}]
        self._precompute_eval_config()
        if self.is_FC and not self.is_OSS:
            obs = self.init_single_multi_turn_FC_entry(0)
            self.step_function = self.step_single_multi_turn_FC_entry
        else:
            obs = self.init_single_multi_turn_entry(0)
            self.step_function = self.step_single_multi_turn_entry
        return [obs], [{"test_id": self.entry_id, "test_category": self.entry_category}]

    def _load_next_sub_entry(self, idx: int):
        """切换到下一个 sub_entry，返回其初始 observation"""
        self.current_memory_entry_idx += 1
        sub_entry = self.memory_group_entries[self.current_memory_entry_idx]
        return self._reset_for_sub_entry(sub_entry)

    def step_single_entry(self, idx: int, action: str):
        # ----------------------------------------
        # memory 专用: prereq/test 串行 done 逻辑
        # ----------------------------------------
        if self.dones[idx]:
            return (
                "Finished,just response ok.",
                0.0,
                self.dones[idx],
                {"test_category": self.entry_category},
            )

        observation = self.step_function(idx, action)

        if self.turns[idx] >= len(self.all_multi_turn_messages):
            is_prereq = is_memory_prereq(self.entry_id)

            if is_prereq:
                # ========== prereq 结束: 先 flush memory 到磁盘，再切换到对应 test ==========
                # base_handler.inference 会在对话结束时 flush，但 BFCLEnv 逐 turn 执行，需在此手动调用
                for class_name in self.involved_classes:
                    instance_name = f"{self.handler.model_name_underline_replaced}_{self.entry_id}_{class_name}_env{self.env_id}_idx{idx}_instance"
                    instance_name = re.sub(r"[-./]", "_", instance_name)
                    if instance_name in globals():
                        memory_instance = globals()[instance_name]
                        if hasattr(memory_instance, "_flush_memory_to_local_file"):
                            memory_instance._flush_memory_to_local_file()
                self.dones[idx] = False
                obs_list, info = self._load_next_sub_entry(idx)
                return obs_list[0], 0.0, False, info

            else:
                # test case 结束: 评估打分，done 取决于是否最后一个 test
                if not self.is_multi_turn_interaction:
                    self.all_model_response[idx] = self.all_model_response[idx][0][0]
                    self.all_reasoning_content[idx] = self.all_reasoning_content[idx][0][0]
                eval_result = self.evaluate_single(idx)
                reward = 1.0 if eval_result.get("valid", False) else 0.0
                self.memory_test_scores.append(reward)
                print(f"Idx: {idx}, Id: {self.entry_id} Finished, Reward: {reward}, Error: {eval_result.get('error', [])}")

                is_last_sub = self.current_memory_entry_idx >= len(self.memory_group_entries) - 1
                if is_last_sub:
                    self.dones[idx] = True
                    return observation, reward, True, eval_result
                else:
                    self.dones[idx] = False
                    obs_list, info = self._load_next_sub_entry(idx)
                    return obs_list[0], reward, False, eval_result

        return (
            observation,
            0.0,
            self.dones[idx],
            {"test_category": self.entry_category},
        )
