"""
BFCL Environment for Reinforcement Learning

该模块将BFCL (Berkeley Function Calling Leaderboard) 封装为RL环境,
复用BFCL原有的评测逻辑,支持所有BFCL任务类别。
"""

import copy
import importlib
import inspect
import json
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING

import bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker as multi_turn_checker
from bfcl_eval.constants.default_prompts import MAXIMUM_STEP_LIMIT
from bfcl_eval.constants.executable_backend_config import (
    CLASS_FILE_PATH_MAPPING,
    STATELESS_CLASSES,
)
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
    is_empty_execute_response,
)

if TYPE_CHECKING:
    from bfcl_eval.eval_checker.multi_turn_eval.func_source_code.memory_api_metaclass import (
        MemoryAPI,
    )

from bfcl_eval.constants.default_prompts import (
    DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_PROMPTING,
)
from bfcl_eval.constants.enums import Language, ReturnFormat
from bfcl_eval.eval_checker.eval_runner import (
    _evaluate_single_agentic_entry,
    _evaluate_single_ast_entry,
    _evaluate_single_multi_turn_entry,
    _evaluate_single_relevance_entry,
)
from bfcl_eval.model_handler.base_handler import add_memory_instruction_system_prompt
from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler
from bfcl_eval.model_handler.utils import parse_prompt_variation_params
from bfcl_eval.utils import (
    contain_multi_turn_interaction,
    extract_test_category_from_id,
    is_agentic,
    is_format_sensitivity,
    is_java,
    is_js,
    is_memory,
    is_multi_turn,
    is_relevance_or_irrelevance,
)


@dataclass
class Choice:
    text: str

    def to_dict(self):
        return {"text": self.text}

    @classmethod
    def from_dict(cls, data: dict):
        return cls(text=data.get("text", ""))


@dataclass
class Usage:
    prompt_tokens: int
    completion_tokens: int

    def to_dict(self):
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            prompt_tokens=data.get("prompt_tokens", 0),
            completion_tokens=data.get("completion_tokens", 0),
        )


@dataclass
class ApiResponse:
    choices: list[Choice]
    usage: Usage

    def to_dict(self):
        return {
            "choices": [choice.to_dict() for choice in self.choices],
            "usage": self.usage.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            choices=[Choice.from_dict(choice) for choice in data.get("choices", [])],
            usage=Usage.from_dict(data.get("usage", {})),
        )


def execute_multi_turn_func_call(
    func_call_list: list[str],
    initial_config: dict,
    involved_classes: list,
    model_name: str,
    test_entry_id: str,
    long_context: bool = False,
    is_evaL_run: bool = False,
    env_id: int = -1,
    idx: int = -1,
) -> tuple[list[str], dict]:
    """
    重写的 execute_multi_turn_func_call，为每个 idx 创建独立实例
    避免 group_n > 1 时不同 idx 之间的状态相互影响
    """
    if is_evaL_run:
        model_name += "_eval"

    class_method_name_mapping = {}
    involved_instances = {}
    for class_name in involved_classes:
        module_name = CLASS_FILE_PATH_MAPPING[class_name]
        # 在实例名中加入 idx，确保每个 idx 有独立的实例
        if env_id == -1 and idx == -1:
            instance_name = f"{model_name}_{test_entry_id}_{class_name}_instance"
        else:
            instance_name = f"{model_name}_{test_entry_id}_{class_name}_env{env_id}_idx{idx}_instance"
        instance_name = re.sub(r"[-./]", "_", instance_name)

        # 使用 globals() 存储，但实例名包含 idx，确保每个 idx 有独立实例
        # 这样 eval 可以正常访问实例，同时避免 group_n > 1 时相互影响
        if instance_name not in globals():
            module = importlib.import_module(module_name)
            class_ = getattr(module, class_name)
            class_instance = class_()
            if class_name not in STATELESS_CLASSES:
                class_initial_config = initial_config.get(class_name, {})
                # Deep copy the initial configuration to avoid mutation issues
                class_instance._load_scenario(copy.deepcopy(class_initial_config), long_context=long_context)
            globals()[instance_name] = class_instance
        else:
            class_instance = globals()[instance_name]

        involved_instances[class_name] = class_instance

        # Retrieve all method names and map them to the instance
        for method_name, method in inspect.getmembers(class_instance, predicate=inspect.ismethod):
            # Skip private methods
            if method_name.startswith("_"):
                continue
            class_method_name_mapping[method_name] = instance_name

    execution_results = []
    for func_call in func_call_list:
        # Add the instance name to the method calls
        func_call = process_method_calls(func_call, class_method_name_mapping)

        # Evaluate the function call
        try:
            # We need to make a copy here because otherwise the `eval(func_call)` would error.
            func_call_copy = func_call
            # Before calling `eval`, we need to make sure that the function call is safe
            # We do so by checking if the function is `kill` or `exit`, etc.
            # Extract the function name first
            if "(" in func_call_copy:
                func_call_copy = func_call_copy.split("(")[0]
            # Situation where the function call is a method call
            if "." in func_call_copy:
                func_call_copy = func_call_copy.split(".")[1]
            if func_call_copy in [
                "kill",
                "exit",
                "quit",
                "remove",
                "unlink",
                "popen",
                "Popen",
                "run",
            ]:
                raise Exception(f"Function call {func_call_copy} is not allowed.")

            func_call_result = eval(func_call)

            if isinstance(func_call_result, str):
                pass
            elif isinstance(func_call_result, dict):
                # Some function returns a object instance, which is not serializable
                try:
                    func_call_result = json.dumps(func_call_result)
                except Exception:
                    func_call_result = str(func_call_result)
            else:
                func_call_result = str(func_call_result)

            execution_results.append(func_call_result)
        except Exception as e:
            execution_results.append(f"Error during execution: {str(e)}")

    return execution_results, involved_instances


def process_method_calls(function_call_string: str, instance_mapping: dict) -> str:
    """
    Prepends the instance name to the function name for each of the function name represented in the string.
    """

    def replace_function(match):
        func_name = match.group(1)
        if func_name in instance_mapping:
            return f"{instance_mapping[func_name]}.{func_name}"
        return func_name

    # Regular expression to match function names
    pattern = r"\b([a-zA-Z_]\w*)\s*(?=\()"

    # Replace function names with their class-prepended versions
    processed_string = re.sub(pattern, replace_function, function_call_string)

    return processed_string



multi_turn_checker.execute_multi_turn_func_call = execute_multi_turn_func_call


class BFCLEnv:
    def __init__(
        self,
        group_n,
        env_id,
        model_name=None,
        handler=None,
        ground_truth_dict=None,
    ):
        self.group_n = group_n
        self.env_id = env_id
        self.model_name = model_name
        self.handler = handler
        self.ground_truth_dict = ground_truth_dict or {}

    def reset(self, entry):
        entry , eval_entry = entry # In simple_java , simple_javascrips test categories,entry and eval_entry are not the same
        self.entries = [deepcopy(entry) for _ in range(self.group_n)]
        self.eval_entries = [deepcopy(eval_entry) for _ in range(self.group_n)]
        self.entry_id = entry["id"]
        self.entry_category = extract_test_category_from_id(self.entry_id)
        ground_truth_id = self.entry_id.split("classic:")[-1] if self.entry_category == "format_sensitivity" else self.entry_id
        self.ground_truth = self.ground_truth_dict.get(ground_truth_id, {})
        self.is_multi_turn_interaction = contain_multi_turn_interaction(self.entry_id)
        self.is_FC = self.handler.is_fc_model if self.handler else False
        self.is_OSS = isinstance(self.handler, OSSHandler) if self.handler else False
        if self.is_multi_turn_interaction:
            self.initial_config = entry.get("initial_config", {})
            self.involved_classes = entry["involved_classes"]
            self.holdout_function = entry.get("missed_function", {})
        else:
            self.initial_config = {}
            self.involved_classes = []
            self.holdout_function = {}

        self.all_multi_turn_messages = entry["question"]

        self.all_model_response = [[[] for _ in range(len(self.all_multi_turn_messages))] for _ in range(self.group_n)]
        self.all_reasoning_content = [[[] for _ in range(len(self.all_multi_turn_messages))] for _ in range(self.group_n)]

        self.turns = [0] * self.group_n
        self.steps = [0] * self.group_n
        self.dones = [False] * self.group_n
        self.inference_data = [{}] * self.group_n
        self._precompute_eval_config()
        observations = []

        if self.is_FC and not self.is_OSS:
            if self.is_multi_turn_interaction:
                observations = [self.init_single_multi_turn_FC_entry(idx) for idx in range(self.group_n)]
                self.step_function = self.step_single_multi_turn_FC_entry
            else:
                observations = [self.init_single_single_turn_FC_entry(idx) for idx in range(self.group_n)]
                self.step_function = self.step_single_single_turn_FC_entry
        else:
            if self.is_multi_turn_interaction:
                observations = [self.init_single_multi_turn_entry(idx) for idx in range(self.group_n)]
                self.step_function = self.step_single_multi_turn_entry
            else:
                observations = [self.init_single_single_turn_entry(idx) for idx in range(self.group_n)]
                self.step_function = self.step_single_single_turn_entry
        return observations, [{"test_id": self.entry_id,"group_id":i, "test_category": self.entry_category} for i in range(self.group_n)]
    def _precompute_eval_config(self):
        """
        预计算评估配置,在reset阶段调用

        根据当前测试用例的类别,预先判断:
        - 评估函数类型 (relevance/agentic/multi_turn/format_sensitivity/ast)
        - 语言类型 (Python/Java/JavaScript)
        - 返回格式 (ReturnFormat)
        - 其他特殊参数 (如format_sensitivity的配置)

        注意: 同一个group共享相同的配置
        """
        test_category = self.entry_category
        test_id = self.entry_id

        config = {
            "test_category": test_category,
            "test_id": test_id,
        }

        # 判断评估类型
        if is_relevance_or_irrelevance(test_category):
            config["eval_type"] = "relevance"

        elif is_agentic(test_category):
            config["eval_type"] = "agentic"

        elif is_multi_turn(test_category):
            config["eval_type"] = "multi_turn"

        elif is_format_sensitivity(test_category):
            config["eval_type"] = "format_sensitivity"
            config["language"] = Language.PYTHON

            # 解析format sensitivity配置
            if ":" in test_id and len(test_id.split(":")) == 3:
                format_sensitivity_config = test_id.split(":")[1]
                try:
                    (
                        return_format_str,
                        has_tool_call_tag,
                        function_doc_format,
                        prompt_format,
                        prompt_style,
                    ) = parse_prompt_variation_params(format_sensitivity_config)
                    config["return_format"] = ReturnFormat(return_format_str)
                    config["has_tool_call_tag"] = has_tool_call_tag
                except Exception:
                    config["return_format"] = ReturnFormat.PYTHON
                    config["has_tool_call_tag"] = False
            else:
                config["return_format"] = ReturnFormat.PYTHON
                config["has_tool_call_tag"] = False

        else:
            # 单轮AST测试
            config["eval_type"] = "ast"

            # 确定语言和返回格式
            if is_java(test_category):
                config["language"] = Language.JAVA
                config["return_format"] = ReturnFormat.JAVA
            elif is_js(test_category):
                config["language"] = Language.JAVASCRIPT
                config["return_format"] = ReturnFormat.JAVASCRIPT
            else:
                config["language"] = Language.PYTHON
                config["return_format"] = ReturnFormat.PYTHON

            config["has_tool_call_tag"] = False

        self.eval_config = config

    def evaluate_single(self, index):
        if index >= self.group_n:
            return {
                "valid": False,
                "test_category": self.entry_category,
                "error": [f"Index {index} out of range (group_n={self.group_n})"],
                "error_type": "index_error",
            }

        if self.eval_config is None:
            return {
                "valid": False,
                "test_category": self.entry_category,
                "error": ["Evaluation config not initialized. Call reset() first."],
                "error_type": "config_error",
            }

        try:
            eval_type = self.eval_config["eval_type"]
            if eval_type == "relevance":
                result = _evaluate_single_relevance_entry(
                    handler=self.handler,
                    index=self.entry_id,
                    model_result_item=self.all_model_response[index],
                    prompt_entry=self.eval_entries[index],
                    model_name=self.model_name,
                    test_category=self.entry_category,
                )

            elif eval_type == "agentic":
                result = _evaluate_single_agentic_entry(
                    handler=self.handler,
                    index=self.entry_id,
                    model_result_list=self.all_model_response[index],
                    possible_answer_item=self.ground_truth.get("ground_truth", None),
                    prompt_entry=self.eval_entries[index],
                    model_name=self.model_name,
                    test_category=self.entry_category,
                )

            elif eval_type == "multi_turn":
                result = _evaluate_single_multi_turn_entry(
                    handler=self.handler,
                    test_entry_id=self.entry_id,
                    model_result_list=self.all_model_response[index],
                    ground_truth_list=self.ground_truth.get("ground_truth", []),
                    prompt_entry=self.eval_entries[index],
                    model_name=self.model_name,
                    test_category=self.entry_category,
                )

            elif eval_type == "format_sensitivity":
                result = _evaluate_single_ast_entry(
                    handler=self.handler,
                    index=self.entry_id,
                    model_result_item=self.all_model_response[index],
                    possible_answer_item=self.ground_truth["ground_truth"],
                    prompt_entry=self.eval_entries[index],
                    model_name=self.model_name,
                    test_category=self.entry_category,
                    language=self.eval_config["language"],
                    return_format=self.eval_config["return_format"],
                    has_tool_call_tag=self.eval_config["has_tool_call_tag"],
                )

            else:
                result = _evaluate_single_ast_entry(
                    handler=self.handler,
                    index=self.entry_id,
                    model_result_item=self.all_model_response[index],
                    possible_answer_item=self.ground_truth["ground_truth"],
                    prompt_entry=self.eval_entries[index],
                    model_name=self.model_name,
                    test_category=self.entry_category,
                    language=self.eval_config["language"],
                    return_format=self.eval_config["return_format"],
                    has_tool_call_tag=self.eval_config["has_tool_call_tag"],
                )

            # 及时清理globals中的实例
            # if result.get('valid', False):
            #    import pdb
            #    pdb.set_trace()

            for class_name in self.involved_classes:
                # eval run时，不会传入env_id和idx，不同idx需要隔离,所以需要清理
                eval_instance_name = f"{self.model_name}_eval_{self.entry_id}_{class_name}_instance"
                ground_truth_eval_instance_name = f"{self.model_name}_ground_truth_eval_{self.entry_id}_{class_name}_instance"
                train_instance_name = f"{self.handler.model_name_underline_replaced}_{self.entry_id}_{class_name}_env{self.env_id}_idx{index}_instance"
                for instance_name in [
                    eval_instance_name,
                    ground_truth_eval_instance_name,
                    train_instance_name,
                ]:
                    instance_name = re.sub(r"[-./]", "_", instance_name)
                    if instance_name in globals():
                        del globals()[instance_name]

            result["test_category"] = self.entry_category

            return result

        except Exception as e:
            # 捕获任何异常并返回错误结果
            import traceback

            error_trace = traceback.format_exc()
            return {
                "id": self.entry_id,
                "model_name": self.model_name,
                "test_category": self.entry_category,
                "valid": False,
                "error": [f"Evaluation failed: {str(e)}"],
                "error_type": "evaluation_error",
                "error_trace": error_trace,
                "prompt": self.eval_entries[index],
                "model_result": self.all_model_response[index],
                "possible_answer": self.ground_truth,
            }

    def init_single_entry(self, idx: int):
        if ("FC" in self.handler.registry_name or self.handler.is_fc_model) and not self.is_OSS:
            if contain_multi_turn_interaction(self.entry_id):
                return self.init_single_multi_turn_FC_entry(idx)
            else:
                return self.init_single_single_turn_FC_entry(idx)
        else:
            if contain_multi_turn_interaction(self.entry_id):
                return self.init_single_multi_turn_entry(idx)
            else:
                return self.init_single_single_turn_entry(idx)

    def init_single_multi_turn_FC_entry(self, idx: int):
        long_context_flag = "long_context" in self.entry_category or "composite" in self.entry_category
        if "long_context" in self.entry_category:
            print(f"DEBUG: entry_id={self.entry_id}, entry_category={self.entry_category}, long_context={long_context_flag}")

        _, involved_instances = execute_multi_turn_func_call(
            func_call_list=[],
            initial_config=self.initial_config,
            involved_classes=self.involved_classes,
            model_name=self.handler.model_name_underline_replaced,
            test_entry_id=self.entry_id,
            long_context=long_context_flag,
            is_evaL_run=False,
            env_id=self.env_id,
            idx=idx,
        )

        if is_memory(self.entry_category):
            assert len(involved_instances) == 1, "Memory category should only involve one class."

            memory_instance: MemoryAPI = list(involved_instances.values())[0]
            self.entries[idx]["question"] = add_memory_instruction_system_prompt(
                self.entries[idx]["question"],
                self.entry_category,
                self.entries[idx]["scenario"],
                memory_instance,
            )
        self.inference_data[idx] = self.handler._pre_query_processing_FC(self.inference_data[idx], self.entries[idx])
        self.inference_data[idx] = self.handler._compile_tools(self.inference_data[idx], self.entries[idx])
        self.prepare_next_turn_inference_data(idx)
        self.all_multi_turn_messages = self.entries[idx]["question"]
        return self.get_obserbation(self.inference_data[idx])

    def init_single_single_turn_entry(self, idx: int):
        self.inference_data[idx] = self.handler._pre_query_processing_prompting(self.entries[idx])
        self.inference_data[idx] = self.handler.add_first_turn_message_prompting(self.inference_data[idx], self.entries[idx]["question"][0])
        return self.get_obserbation(self.inference_data[idx])

    def init_single_single_turn_FC_entry(self, idx: int):
        self.inference_data[idx] = self.handler._pre_query_processing_FC(self.inference_data[idx], self.entries[idx])
        self.inference_data[idx] = self.handler._compile_tools(self.inference_data[idx], self.entries[idx])
        self.inference_data[idx] = self.handler.add_first_turn_message_FC(self.inference_data[idx], self.entries[idx]["question"][0])
        return self.get_obserbation(self.inference_data[idx])

    def init_single_multi_turn_entry(self, idx: int):
        long_context_flag = "long_context" in self.entry_category or "composite" in self.entry_category
        if "long_context" in self.entry_category:
            print(f"DEBUG: entry_id={self.entry_id}, entry_category={self.entry_category}, long_context={long_context_flag}")

        _, involved_instances = execute_multi_turn_func_call(
            func_call_list=[],
            initial_config=self.initial_config,
            involved_classes=self.involved_classes,
            model_name=self.handler.model_name_underline_replaced,
            test_entry_id=self.entry_id,
            long_context=long_context_flag,
            is_evaL_run=False,
            env_id=self.env_id,
            idx=idx,
        )

        if is_memory(self.entry_category):
            assert len(involved_instances) == 1, "Memory category should only involve one class."

            memory_instance: MemoryAPI = list(involved_instances.values())[0]
            self.entries[idx]["question"] = add_memory_instruction_system_prompt(
                self.entries[idx]["question"],
                self.entry_category,
                self.entries[idx]["scenario"],
                memory_instance,
            )
        self.inference_data[idx] = self.handler._pre_query_processing_prompting(self.entries[idx])
        self.all_multi_turn_messages = self.entries[idx]["question"]
        self.prepare_next_turn_inference_data(idx)
        return self.get_obserbation(self.inference_data[idx])

    def step_single_single_turn_entry(self, idx, action: str):

        api_response = self.wrap_as_api_response(action)
        model_response_data = self.handler._parse_query_response_prompting(api_response)
        model_response = model_response_data["model_responses"]
        self.inference_data[idx] = self.handler._add_assistant_message_prompting(self.inference_data[idx], model_response_data)
        self.all_model_response[idx][self.turns[idx]].append(model_response)
        reasoning_content = model_response_data.get("reasoning_content", "")
        self.all_reasoning_content[idx][self.turns[idx]].append(reasoning_content)
        self.turns[idx] += 1
        observation = self.get_obserbation(self.inference_data[idx])
        return observation

    def step_single_multi_turn_entry(self, idx: int, action: str):
        print(f"Idx: {idx}, Id: {self.entry_id}, Turn: {self.turns[idx]}, Step: {self.steps[idx]},total turns: {len(self.all_multi_turn_messages)}")
        api_response = self.wrap_as_api_response(action)
        model_response_data = self.handler._parse_query_response_prompting(api_response)
        model_response = model_response_data["model_responses"]
        self.inference_data[idx] = self.handler._add_assistant_message_prompting(self.inference_data[idx], model_response_data)
        self.all_model_response[idx][self.turns[idx]].append(model_response)
        reasoning_content = model_response_data.get("reasoning_content", "")
        self.all_reasoning_content[idx][self.turns[idx]].append(reasoning_content)
        goto_next_turn = False
        try:
            decoded_model_responses = self.handler.decode_execute(model_response, has_tool_call_tag=False)
            model_response_data["model_responses_decoded"] = decoded_model_responses
            if is_empty_execute_response(decoded_model_responses):
                print("Empty response from the model. Proceed to next turn.")
                goto_next_turn = True
            long_context_flag = "long_context" in self.entry_category or "composite" in self.entry_category
            execution_results, involved_instances = execute_multi_turn_func_call(
                func_call_list=decoded_model_responses,
                initial_config=self.initial_config,
                involved_classes=self.involved_classes,
                model_name=self.handler.model_name_underline_replaced,
                test_entry_id=self.entry_id,
                long_context=long_context_flag,
                is_evaL_run=False,
                env_id=self.env_id,
                idx=idx,
            )
            self.inference_data[idx] = self.handler._add_execution_results_prompting(self.inference_data[idx], execution_results, model_response_data)
        except Exception as e:
            print(f"Error decoding the model response: {e}. Proceed to next turn.")
            goto_next_turn = True

        self.steps[idx] += 1
        if self.steps[idx] >= MAXIMUM_STEP_LIMIT:
            self.turns[idx] = len(self.all_multi_turn_messages)

        if goto_next_turn:
            self.turns[idx] += 1
            self.prepare_next_turn_inference_data(idx)

        observation = self.get_obserbation(self.inference_data[idx])
        return observation

    def step_single_multi_turn_FC_entry(self, idx: int, action: str):
        print(f"Idx: {idx}, Id: {self.entry_id}, Turn: {self.turns[idx]}, Step: {self.steps[idx]},total turns: {len(self.all_multi_turn_messages)}")
        api_response = self.wrap_as_api_response(action)
        model_response_data = self.handler._parse_query_response_FC(api_response)
        model_response = model_response_data["model_responses"]
        self.inference_data[idx] = self.handler._add_assistant_message_FC(self.inference_data[idx], model_response_data)
        self.all_model_response[idx][self.turns[idx]].append(model_response)
        reasoning_content = model_response_data.get("reasoning_content", "")
        self.all_reasoning_content[idx][self.turns[idx]].append(reasoning_content)
        goto_next_turn = False
        try:
            decoded_model_responses = self.handler.decode_execute(model_response, has_tool_call_tag=False)
            model_response_data["model_responses_decoded"] = decoded_model_responses
            if is_empty_execute_response(decoded_model_responses):
                print("Empty response from the model. Proceed to next turn.")
                goto_next_turn = True
            long_context_flag = "long_context" in self.entry_category or "composite" in self.entry_category
            execution_results, involved_instances = execute_multi_turn_func_call(
                func_call_list=decoded_model_responses,
                initial_config=self.initial_config,
                involved_classes=self.involved_classes,
                model_name=self.handler.model_name_underline_replaced,
                test_entry_id=self.entry_id,
                long_context=long_context_flag,
                is_evaL_run=False,
                env_id=self.env_id,
                idx=idx,
            )

            self.inference_data[idx] = self.handler._add_execution_results_FC(self.inference_data[idx], execution_results, model_response_data)
        except Exception as e:
            print(f"Error decoding the model response: {e}")
            goto_next_turn = True

        self.steps[idx] += 1
        if self.steps[idx] >= MAXIMUM_STEP_LIMIT:
            self.turns[idx] = len(self.all_multi_turn_messages)

        if goto_next_turn:
            self.turns[idx] += 1
            self.prepare_next_turn_inference_data(idx)

        observation = self.get_obserbation(self.inference_data[idx])
        return observation

    def step_single_single_turn_FC_entry(self, idx: int, action: str):
        api_response = self.wrap_as_api_response(action)
        model_response_data = self.handler._parse_query_response_FC(api_response)
        model_response = model_response_data["model_responses"]
        self.inference_data[idx] = self.handler._add_assistant_message_FC(self.inference_data[idx], model_response_data)
        self.all_model_response[idx][self.turns[idx]].append(model_response)
        reasoning_content = model_response_data.get("reasoning_content", "")
        self.all_reasoning_content[idx][self.turns[idx]].append(reasoning_content)
        self.turns[idx] += 1
        observation = self.get_obserbation(self.inference_data[idx])
        return observation

    def step_single_entry(self, idx: int, action: str):
        if self.dones[idx]:
            return (
                "Finished,just response ok.",
                0.0,
                self.dones[idx],
                {"test_id": self.entry_id,"group_id":idx, "test_category": self.entry_category},
            )

        observation = self.step_function(idx, action)
        if self.turns[idx] >= len(self.all_multi_turn_messages):
            self.dones[idx] = True
            if not self.is_multi_turn_interaction:
                self.all_model_response[idx] = self.all_model_response[idx][0][0]
                self.all_reasoning_content[idx] = self.all_reasoning_content[idx][0][0]
            eval_result = self.evaluate_single(idx)
            reward = 1.0 if eval_result.get("valid", False) else 0.0
            error = eval_result.get("error", [])
            eval_result["group_id"] = idx
            print(f"Idx: {idx}, Id: {self.entry_id} Finished, Reward: {reward}, Error: {error}")
            return observation, reward, self.dones[idx], eval_result
        else:
            return (
                observation,
                0.0,
                self.dones[idx],
                {"test_id": self.entry_id,"group_id":idx, "test_category": self.entry_category},
            )

    def get_obserbation(self, inference_data: dict):
        function: list[dict] = inference_data["function"]
        message: list[dict] = inference_data["message"]

        formatted_prompt: str = self.handler._format_prompt(message, function)
        return formatted_prompt

    def wrap_as_api_response(self, model_response: str):
        return ApiResponse(
            choices=[Choice(text=model_response)],
            usage=Usage(prompt_tokens=0, completion_tokens=0),
        )

    def prepare_next_turn_inference_data(self, idx: int):
        # 获取当前turn的消息
        if self.turns[idx] < len(self.all_multi_turn_messages):
            next_turn_message = self.all_multi_turn_messages[self.turns[idx]]
        else:
            next_turn_message = []

        # 处理holdout function的情况
        if str(self.turns[idx]) in self.holdout_function:
            assert len(next_turn_message) == 0, "Holdout turn should not have user message."
            next_turn_message = [
                {
                    "role": "user",
                    "content": DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_PROMPTING.format(functions=self.holdout_function[str(self.turns[idx])]),
                }
            ]

        if self.turns[idx] == 0:
            self.inference_data[idx] = self.handler.add_first_turn_message_prompting(self.inference_data[idx], next_turn_message)
        else:
            self.inference_data[idx] = self.handler._add_next_turn_user_message_prompting(self.inference_data[idx], next_turn_message)
