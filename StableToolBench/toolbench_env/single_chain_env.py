import asyncio
from copy import deepcopy

from toolbench.inference.Tree.Tree import tree_node

from .base_env import ToolbenchEnv


class SingleChainEnv(ToolbenchEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def to_json(self, answer=False, process=True):
        if process:
            json_obj = {
                "win": self.status == 1,
                "try_count": len(self.try_list),
                "trys": self.try_list,
                "compare_candidates": [],
                "forward_args": self.forward_args,
            }
            for node in self.terminal_node:
                if not node.pruned:
                    json_obj["compare_candidates"].append(node.get_chain_result_from_this_node(use_messages=False))
        else:
            json_obj = {}

        if answer:
            json_obj["answer_generation"] = {
                "valid_data": False,
                "final_answer": "",
                "function": self.io_func.functions,
                "query_count": self.query_count,
                "total_tokens": self.total_tokens,
                "train_messages": [],
                "chain": [],
            }
            for node in self.terminal_node:
                if not node.pruned:
                    json_obj["answer_generation"]["valid_data"] = True
                    json_obj["answer_generation"]["final_answer"] = node.description
                    json_obj["answer_generation"]["train_messages"] = node.get_train_messages_from_this_node()
                    break
        return json_obj

    async def step(self, action):
        """
        Step 内部为异步，可以并发执行 HTTP 请求，但 Evaluate 部分会排队。
        """
        if self.dones:
            return (
                "Done,just say Finished",
                0.0,
                self.dones,
                {"test_category": "ToolBench"},
            )
        new_message, error_code, total_tokens = self.action_to_message(action)
        self.total_tokens += total_tokens
        self.query_count += 1
        assert new_message["role"] == "assistant"

        if "content" in new_message.keys() and new_message["content"] is not None:
            temp_node = tree_node()
            temp_node.node_type = "Thought"
            temp_node.description = new_message["content"]
            child_io_state = deepcopy(self.now_node.io_state)

            temp_node.io_state = child_io_state
            temp_node.is_terminal = child_io_state.check_success() != 0
            temp_node.messages = self.now_node.messages.copy()
            temp_node.father = self.now_node
            self.now_node.children.append(temp_node)
            temp_node.print(self.process_id)
            self.now_node = temp_node

            if error_code != 0:
                self.now_node.observation_code = error_code
                self.now_node.pruned = True

        if "tool_calls" in new_message.keys() and new_message["tool_calls"] is not None and len(new_message["tool_calls"]) > 0:
            tool_calls = new_message["tool_calls"]
            if self.process_id == 0:
                print("number of parallel calls:", len(tool_calls))

            for i in range(len(tool_calls)):
                function_name = tool_calls[i]["function"]["name"]
                temp_node = tree_node()
                temp_node.node_type = "Action"
                temp_node.description = function_name
                child_io_state = deepcopy(self.now_node.io_state)

                temp_node.io_state = child_io_state
                temp_node.is_terminal = child_io_state.check_success() != 0
                temp_node.messages = self.now_node.messages.copy()
                temp_node.father = self.now_node
                self.now_node.children.append(temp_node)

                # temp_node.print(self.process_id)
                self.now_node = temp_node

                function_input = tool_calls[i]["function"]["arguments"]
                temp_node = tree_node()
                temp_node.node_type = "Action Input"
                temp_node.description = function_input
                child_io_state = deepcopy(self.now_node.io_state)

                # ================= 异步并发调用 API =================
                # 获取当前运行的 loop（即 VectorEnv 里的 self.loop）
                loop = asyncio.get_running_loop()
                try:
                    observation, status = await loop.run_in_executor(
                        None,
                        lambda child_io_state=child_io_state, function_input=function_input, action_name=self.now_node.description: child_io_state.step(
                            action_name=action_name,
                            action_input=function_input,
                        ),
                    )
                except Exception as e:
                    # 将底层执行错误统一视为“无对应 API 名称”（status=1），
                    # 与 rapidapi_wrapper._step 返回 code=1 的语义保持一致，
                    # 允许上层环境按“工具名幻觉”逻辑继续尝试，而不是直接中止整个进程。
                    print(f"\033[93m Error in tool call (process_id={self.process_id}): {e}\033[0m")
                    observation = f"{{\"error\": \"Exception in tool call: {e}\", \"response\": \"\"}}"
                    status = 1
                # ===================================================

                temp_node.observation = observation
                temp_node.observation_code = status

                temp_node.io_state = child_io_state
                temp_node.is_terminal = child_io_state.check_success() != 0
                temp_node.messages = self.now_node.messages.copy()
                temp_node.father = self.now_node
                self.now_node.children.append(temp_node)
                # temp_node.print(self.process_id)
                self.now_node = temp_node

                if status != 0:
                    if status == 4:
                        self.now_node.pruned = True
                    elif status == 1:
                        assert "tool_calls" in new_message.keys() and len(new_message["tool_calls"]) > 0
                        tool_calls[i]["function"]["name"] = "invalid_hallucination_function_name"

                if i == 0:
                    self.now_node.messages.append(new_message)
                if self.now_node.node_type == "Action Input":
                    self.now_node.messages.append(
                        {
                            "role": "tool",
                            "name": tool_calls[i]["function"]["name"],
                            "content": self.now_node.observation,
                            "tool_call_id": tool_calls[i]["id"],
                        }
                    )
        else:
            self.now_node.messages.append(new_message)

        if self.now_node.get_depth() >= self.single_chain_max_step and not (self.now_node.is_terminal):
            self.now_node.pruned = True

        if self.now_node.pruned or self.now_node.is_terminal:
            self.terminal_node.append(self.now_node)
            self.try_list.append(self.to_json_single())

            await self._finish_and_evaluate()

            return (
                self.history_to_observation(),
                self.score,
                self.dones,
                {"valid": self.score > 0, "test_category": "ToolBench"},
            )

        return (
            self.history_to_observation(),
            0.0,
            self.dones,
            {"test_category": "ToolBench"},
        )
