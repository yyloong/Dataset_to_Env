import asyncio
import json
import random
from copy import deepcopy

from toolbench.inference.Prompts.Tree_search_prompts import DIVERSITY_PROMPT
from toolbench.inference.Tree.Tree import tree_node

from .base_env import ToolbenchEnv


class DFSEnv(ToolbenchEnv):
    """Implement of CoT method"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset(self, io_func, query_id):
        self.node_stack = []
        self.now_next_tree_split_node = None
        self.give_up_node = []
        self.now_expand_num = 0
        self._tree_beam_size = self.specific_args.tree_beam_size
        self._max_query_count = self.specific_args.max_query_count
        self._answer = self.specific_args.answer
        self.delete_former_diversity_message = False
        return super().reset(io_func, query_id)

    def to_json(self, answer=False, process=True):

        if process:
            json_obj = {
                "win": self.status == 1,
                "tree": self.tree.to_json_recursive(),
                "forward_args": self.forward_args,
                "compare_candidates": [],
            }
            for node in self.terminal_node:
                if not node.pruned:  # has answer
                    json_obj["compare_candidates"].append(node.get_chain_result_from_this_node(use_messages=False))
        else:
            json_obj = {}

        if answer:
            json_obj["answer_generation"] = {
                "valid_data": False,
                "query_count": self.query_count,
                "total_tokens": self.total_tokens,
                "final_answer": "",
                "finish_type": "give_answer",
                "function": self.io_func.functions,
                "chain": [],
            }
            for node in self.terminal_node:
                if not node.pruned:
                    json_obj["answer_generation"]["valid_data"] = True
                    json_obj["answer_generation"]["finish_type"] = "give_answer"
                    json_obj["answer_generation"]["final_answer"] = node.description
                    json_obj["answer_generation"]["train_messages"] = node.get_train_messages_from_this_node()
                    break
            # do not have final answer, look for give_up
            if not json_obj["answer_generation"]["valid_data"]:
                if len(self.give_up_node) > 0:
                    random_pos = random.randint(0, len(self.give_up_node) - 1)
                    choose_give_up_node = self.give_up_node[random_pos]
                    json_obj["answer_generation"]["valid_data"] = True
                    json_obj["answer_generation"]["finish_type"] = "give_up"
                    json_obj["answer_generation"]["final_answer"] = choose_give_up_node.description
                    json_obj["answer_generation"]["train_messages"] = choose_give_up_node.get_train_messages_from_this_node()
        return json_obj

    def diversity_prompt(self, node):
        """If a node have children now, We will prompt the model to generate different nodes than all the existing nodes"""
        delete_former_diversity_message = False
        diversity_message = None
        if len(node.children) > 0:
            former_candidates_des = ""
            js_list = []
            for k, child in enumerate(node.children):
                temp_node = child
                while not temp_node.is_terminal and temp_node.node_type != "Action Input" and len(temp_node.children) > 0:
                    temp_node = temp_node.children[0]
                if temp_node.node_type == "Action Input":
                    obj_dict = {
                        "name": temp_node.father.description,
                        "arguments": temp_node.description,
                        "function_output": temp_node.observation,
                        "mento-carlo-action-value": temp_node.compute_weight(),
                    }
                    js_list.append(obj_dict)

            if len(js_list) > 0:
                former_candidates_des = former_candidates_des + f"{json.dumps(js_list, indent=2)}\n"
                if node.observation != "":
                    former_candidates_des = former_candidates_des + f"again, your former observation: {node.observation}\n"
                diverse_prompt = DIVERSITY_PROMPT
                diverse_prompt = diverse_prompt.replace("{previous_candidate}", former_candidates_des)
                diversity_message = {"role": "user", "content": diverse_prompt}
                node.messages.append(diversity_message)

                delete_former_diversity_message = True

        return delete_former_diversity_message

    async def step(self, action):
        final_answer_back_length = 2
        prune_back_length = 2
        if self.dones:
            return (
                "Done,just say Finished",
                0.0,
                self.dones,
                {"test_category": "ToolBench"},
            )

        self.now_node.expand_num = self.now_expand_num
        self.now_expand_num += 1

        new_message, error_code, total_tokens = self.action_to_message(action)
        new_message = {k: v for k, v in new_message.items() if v is not None}

        self.total_tokens += total_tokens
        self.query_count += 1
        if self.query_count >= self._max_query_count:
            print(f"Up to max query count: {self._max_query_count}")
            await self._finish_and_evaluate()
            return (
                self.history_to_observation(),
                self.score,
                True,
                {"valid": self.score > 0, "test_category": "ToolBench"},
            )

        if self.delete_former_diversity_message:
            self.now_node.messages[-1]["valid"] = False

        assert new_message["role"] == "assistant"

        expand_node = self.now_node  # 当前扩展点，复用变量避免冲突
        # ---------- 复用 single_chain 的节点构建逻辑 ----------
        if "content" in new_message.keys() and new_message["content"] is not None:
            temp_node = tree_node()
            temp_node.node_type = "Thought"
            temp_node.description = new_message["content"]
            child_io_state = deepcopy(self.now_node.io_state)
            child_io_state.retriever = None

            temp_node.io_state = child_io_state
            temp_node.is_terminal = child_io_state.check_success() != 0
            temp_node.messages = deepcopy(expand_node.messages)
            temp_node.father = expand_node
            expand_node.children.append(temp_node)
            expand_node = temp_node

            if error_code != 0:
                expand_node.observation_code = error_code
                expand_node.pruned = True

        if "tool_calls" in new_message.keys() and new_message["tool_calls"] is not None and len(new_message["tool_calls"]) > 0:
            tool_calls = new_message["tool_calls"]
            if self.process_id == 0:
                print("number of parallel calls:", len(tool_calls))

            for i in range(len(tool_calls)):
                function_name = tool_calls[i]["function"]["name"]
                temp_node = tree_node()
                temp_node.node_type = "Action"
                temp_node.description = function_name
                child_io_state = deepcopy(expand_node.io_state)
                child_io_state.retriever = None

                temp_node.io_state = child_io_state
                temp_node.is_terminal = child_io_state.check_success() != 0
                temp_node.messages = deepcopy(expand_node.messages)
                temp_node.father = expand_node
                expand_node.children.append(temp_node)
                expand_node = temp_node

                function_input = tool_calls[i]["function"]["arguments"]
                temp_node = tree_node()
                temp_node.node_type = "Action Input"
                temp_node.description = function_input
                child_io_state = deepcopy(expand_node.io_state)
                child_io_state.retriever = None

                # ================= 异步并发调用 API =================
                # 获取当前运行的 loop（即 VectorEnv 里的 self.loop）
                loop = asyncio.get_running_loop()
                observation, status = await loop.run_in_executor(
                    None,
                    lambda child_io_state=child_io_state, function_input=function_input, action_name=expand_node.description: child_io_state.step(
                        action_name=action_name,
                        action_input=function_input,
                    ),
                )
                # ===================================================

                temp_node.observation = observation
                temp_node.observation_code = status
                temp_node.io_state = child_io_state
                temp_node.is_terminal = child_io_state.check_success() != 0
                temp_node.messages = deepcopy(expand_node.messages)
                temp_node.father = expand_node
                expand_node.children.append(temp_node)
                expand_node = temp_node

                if status != 0:
                    if status == 4:
                        expand_node.pruned = True
                        self.give_up_node.append(expand_node)
                    elif status == 1:
                        assert "tool_calls" in new_message.keys() and len(new_message["tool_calls"]) > 0
                        tool_calls[i]["function"]["name"] = "invalid_hallucination_function_name"
                    elif status == 3:
                        expand_node.is_terminal = True
                        expand_node.make_finish(final_answer_back_length)

                if i == 0:
                    expand_node.messages.append(new_message)
                if expand_node.node_type == "Action Input":
                    expand_node.messages.append(
                        {
                            "role": "tool",
                            "name": tool_calls[i]["function"]["name"],
                            "content": expand_node.observation,
                            "tool_call_id": tool_calls[i]["id"],
                        }
                    )
                self.node_stack.append(expand_node)
        else:
            expand_node.messages.append(new_message)

        if expand_node.get_depth() >= self.single_chain_max_step or expand_node.pruned or expand_node.is_terminal:
            # import pdb; pdb.set_trace()
            if expand_node.is_terminal:
                self.status = 1
                self.terminal_node.append(expand_node)
                print(f"terminal node: {expand_node.description}")
                for i in range(final_answer_back_length):
                    if not self.node_stack:
                        break
                    pop_node = self.node_stack.pop()
                    pop_node.make_finish(1)
            else:
                expand_node.pruned = True
                if expand_node.observation_code == 4:
                    self.give_up_node.append(expand_node)
                    print(f"give up node: {expand_node.description}")
                    for i in range(prune_back_length):
                        if not self.node_stack:
                            break
                        pop_node = self.node_stack.pop()
                        pop_node.make_finish(1)
                else:
                    print(f"pruned node: {expand_node.description}")
                    pop_node = self.node_stack.pop()
                    pop_node.make_finish(1)

            if len(self.node_stack) > 0:
                self.now_node = self.node_stack[-1]
            if len(self.terminal_node) >= self._answer:
                print(f"Up to max answer count: {self._answer}")
                await self._finish_and_evaluate()
                self.dones = True
                return (
                    self.history_to_observation(),
                    self.score,
                    self.dones,
                    {"valid": self.score > 0, "test_category": "ToolBench"},
                )
        if len(self.node_stack) > 0:
            self.now_node = self.node_stack[-1]
            print(f"expand node: {expand_node.description}")

            self.delete_former_diversity_message = self.diversity_prompt(self.now_node)

            now_finished_children = [c for c in self.now_node.children if getattr(c, "finished", False)]
            if len(now_finished_children) >= self._tree_beam_size:
                print("finished exporation")
                self.node_stack.pop()

        if len(self.node_stack) > 0:
            return (
                self.history_to_observation(),
                0.0,
                self.dones,
                {"test_category": "ToolBench"},
            )

        self.dones = True
        await self._finish_and_evaluate()
        return (
            self.history_to_observation(),
            self.score,
            self.dones,
            {"valid": self.score > 0, "test_category": "ToolBench"},
        )
