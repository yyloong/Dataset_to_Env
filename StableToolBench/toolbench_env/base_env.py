import asyncio
import json
import random
import string
from copy import deepcopy

import backoff
from toolbench.inference.Prompts.ReAct_prompts import (
    FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION,
    FORMAT_INSTRUCTIONS_USER_FUNCTION,
)
from toolbench.inference.Tree.Tree import my_tree
from toolbench.inference.utils import react_parser
from toolbench.model.model_adapter import get_conversation_template
from toolbench.tooleval.convert_to_answer_format import (
    process_invalid_data,
    process_valid_data,
)
from toolbench.tooleval.evaluators import load_registered_automatic_evaluator
from toolbench.tooleval.evaluators.registered_cls.rtl import AnswerStatus
from toolbench.tooleval.utils import get_steps
from toolbench.utils import process_system_message


@backoff.on_exception(backoff.expo, Exception, max_time=15)
def evaluate_answer(evaluator, query_id, example, evaluate_time):
    # 保持原有的同步逻辑
    answer_steps, final_step = get_steps(example)

    if "'name': 'Finish'" not in final_step:
        return query_id, AnswerStatus.Unsolved, evaluate_time

    is_solved, is_solved_reason = evaluator.check_is_solved(
        {
            "query": example["query"],
            "available_tools": example["available_tools"],
        },
        example["answer"],
        return_reason=True,
    )
    return query_id, is_solved, evaluate_time


class ToolbenchEnv:
    def __init__(
        self,
        evaluator_name,
        evaluators_cfg_path,
        extra_prefix="",
        process_id=0,
        group_id=0,
        env_id=0,
        start_message_list=None,
        single_chain_max_step=10,
        evaluation_times=1,
        tokenizer=None,
        eval_lock=None,
        specific_args=None,
    ):
        self.io_func = None
        self.extra_prefix = extra_prefix
        self.process_id = process_id
        self.group_id = group_id
        self.env_id = env_id
        self.single_chain_max_step = single_chain_max_step
        self.evaluator = load_registered_automatic_evaluator(
            evaluator_name=evaluator_name,
            evaluators_cfg_path=evaluators_cfg_path,
        )

        self.evaluation_times = evaluation_times
        self.tokenizer = tokenizer
        self.eval_lock = eval_lock
        self.specific_args = specific_args

    def construct_tree_root(self):
        self.forward_args = locals()
        if "self" in self.forward_args.keys():
            self.forward_args.pop("self")
        self.tree = my_tree()
        self.tree.root.node_type = "Action Input"
        self.tree.root.io_state = deepcopy(self.io_func)
        system = FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION
        system = system.replace("{task_description}", self.io_func.task_description)
        self.tree.root.messages.append({"role": "system", "content": system})
        user = FORMAT_INSTRUCTIONS_USER_FUNCTION
        user = user.replace("{input_description}", self.io_func.input_description)
        self.tree.root.messages.append({"role": "user", "content": user})
        return self.tree.root

    def reset(self, io_func, query_id):
        self.score = 0.0
        self.query_id = query_id
        self.dones = False
        self.io_func = io_func
        self.now_node = None
        self.status = 0
        self.try_list = []
        self.terminal_node = []

        self.query_count = 0
        self.total_tokens = 0
        self.success_count = 0

        self.now_node = self.construct_tree_root()
        return self.history_to_observation(), {"test_category": "ToolBench"}

    def to_json(self, answer=False, process=True):
        raise NotImplementedError("to_json is not implemented")

    def to_json_single(self):
        json_obj = {}
        tree_obj = self.terminal_node[-1].get_chain_result_from_this_node()
        json_obj["chain"] = tree_obj
        json_obj["win"] = self.status == 1
        return json_obj

    def action_to_message(self, action):
        decoded_token_len = len(self.tokenizer(action))
        if decoded_token_len >= 8192:
            message = {
                "role": "assistant",
                "content": "The response is too long, please try again.\nOrignal response: " + action,
                "tool_calls": [
                    {
                        "id": 0,
                        "function": {
                            "name": "Finish",
                            "arguments": json.dumps({"return_type": "give_up_and_restart"}),
                        },
                        "type": "function",
                    }
                ],
            }
            return message, 0, decoded_token_len

        thought, parsed_action, action_input = react_parser(action)
        random_id = "".join([random.choice(string.ascii_letters + string.digits) for _ in range(8)])
        message = {
            "role": "assistant",
            "content": action,
            "tool_calls": [
                {
                    "id": random_id,  # 可能在conver_to_answer_format处有bug
                    "function": {"name": parsed_action, "arguments": action_input},
                    "type": "function",
                }
            ],
        }
        return message, 0, decoded_token_len

    async def step(self, action):
        raise NotImplementedError("step is not implemented")

    def history_to_observation(self):
        conv = get_conversation_template(self.specific_args.template)
        tools = self.io_func.functions
        if tools != []:
            functions = [tool["function"] for tool in tools]

        if self.specific_args.template == "chat_model":
            chat_messages = []
            for message in self.now_node.messages:
                role = message["role"]
                content = message["content"]
                if role == "system":
                    content = process_system_message(content, functions=functions)
                elif role == "tool":
                    content = "Function response:" + content
                    role = "user"
                chat_messages.append({"role": role, "content": content})
            prompt = self.tokenizer.apply_chat_template(chat_messages, add_generation_prompt=True, tokenize=False)
            return prompt

        elif self.specific_args.template == "tool-llama":
            roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
        elif self.specific_args.template == "tool-llama-single-round" or self.specific_args.template == "tool-llama-multi-rounds":
            roles = {
                "system": conv.roles[0],
                "user": conv.roles[1],
                "tool": conv.roles[2],
                "assistant": conv.roles[3],
            }
        else:
            raise ValueError(f"Invalid template: {self.specific_args.template}")

        prompt = ""
        for message in self.now_node.messages:
            role = roles[message["role"]]
            content = message["content"]
            if role == "System" and tools != []:
                content = process_system_message(content, functions=functions)
            prompt += f"{role}: {content}\n"
        prompt += "Assistant:\n"
        return prompt

    def _evaluate_sync(self):
        """
        同步评估函数，包含阻塞的 evaluate_answer 调用
        """
        answer = self.to_json(answer=True, process=True)
        answer["answer_generation"]["query"] = self.io_func.input_description
        if not answer["answer_generation"]["valid_data"]:
            answer = process_invalid_data(self.specific_args.method, answer)
        else:
            answer = process_valid_data(self.specific_args.method, answer["answer_generation"])

        score = 0.0

        for eval_time in range(self.evaluation_times):
            try:
                _, is_solved, _ = evaluate_answer(self.evaluator, self.query_id, answer, eval_time)
            except Exception as e:
                print(f"\033[93m Error: {e}, query_id: {self.query_id}\033[0m")
                continue
            if str(is_solved) == "AnswerStatus.Solved":
                score += 1
            elif str(is_solved) == "AnswerStatus.Unsure":
                score += 0.5

        self.score = score / self.evaluation_times

    async def _finish_and_evaluate(self):
        """结束并串行评估，复用原 step 的评估逻辑。"""
        self.dones = True
        async with self.eval_lock:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._evaluate_sync)
        print(f"\033[93m Process_ID: {self.process_id}, Env_ID: {self.env_id},Group_ID: {self.group_id}, Score: {self.score}, query_id: {self.query_id}\033[0m")
