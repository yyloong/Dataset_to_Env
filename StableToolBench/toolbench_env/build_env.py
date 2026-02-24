import asyncio
import json
import random

from toolbench.inference.Downstream_tasks.rapidapi_multithread import contain, get_white_list, rapidapi_wrapper, standardize
from toolbench.inference.LLM.retriever import ToolRetriever
from transformers import AutoTokenizer

from toolbench_env.DFS_env import DFSEnv
from toolbench_env.single_chain_env import (
    SingleChainEnv,
)


def build_toolbench_envs(
    env_num,
    group_n,
    resources_per_worker,
    specific_args,
    is_train=True,
):
    querys = json.load(open(specific_args.input_query_dir))
    white_list = get_white_list(specific_args.tool_root_dir)

    # 前置过滤：只保留通过 white_list 校验的 query，避免 reset 时数量不一致
    original_query_count = len(querys)
    filtered_querys = []
    for data_dict in querys:
        if "api_list" in data_dict:
            origin_tool_names = [standardize(cont["tool_name"]) for cont in data_dict["api_list"]]
            tool_des = contain(origin_tool_names, white_list)
            if tool_des is False:
                continue
            filtered_querys.append(data_dict)
        else:
            filtered_querys.append(data_dict)
    querys = filtered_querys
    filtered_count = original_query_count - len(querys)
    if filtered_count > 0:
        print(f"\033[93mFiltered {filtered_count} queries (not in white_list). Remaining: {len(querys)} / {original_query_count}\033[0m")
    #######################################################################

    retriever = None
    if specific_args.use_retriever:
        retriever = ToolRetriever(
            corpus_tsv_path=specific_args.corpus_tsv_path,
            model_path=specific_args.retrieval_model_path,
        )
    if is_train:
        print(f"\033[93mTraining mode, querys length: {len(querys)}\033[0m")
    else:
        print(f"\033[93mEvaluation mode, querys length: {len(querys)}\033[0m")
        if len(querys) % env_num != 0:
            # 用黄色打印警告
            print("\033[93mWarning: In evaluation mode, the number of queries is not divisible by the number of environments. The last batch will be truncated.\033[0m")
            print("\033[93mIf you want to use full evaluation data, please set the number of environments to be a multiple of the number of queries.\033[0m")
            querys = querys[: len(querys) - len(querys) % env_num]
            print(f"\033[93mTruncated queries to: {len(querys)}\033[0m")

    tokenizer = AutoTokenizer.from_pretrained(
        specific_args.model_path,
        use_fast=False,
        model_max_length=specific_args.max_sequence_length,
        local_files_only=getattr(specific_args, "local_files_only", False),
        trust_remote_code=True,
    )

    return ToolbenchVectorEnv(env_num, group_n, query_list=querys, is_random=is_train, white_list=white_list, retriever=retriever, specific_args=specific_args, tokenizer=tokenizer, resources_per_worker=resources_per_worker["num_cpus"])


class ToolbenchVectorEnv:
    def __init__(self, env_num, group_n, query_list=None, is_random=None, white_list=None, retriever=None, specific_args=None, tokenizer=None, resources_per_worker=None):
        self.env_num = env_num
        self.group_n = group_n
        self.now_index = 0
        self._query_list = query_list
        self._is_random = is_random
        self._white_list = white_list
        self.retriever = retriever
        self.specific_args = specific_args
        self.tokenizer = tokenizer
        self.resources_per_worker = resources_per_worker
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.eval_lock = asyncio.Lock()
        if self.specific_args.method == "CoT":
            self.env_type = SingleChainEnv
        elif self.specific_args.method == "DFS":
            self.env_type = DFSEnv

        self.envs = [
            self.env_type(
                self.specific_args.evaluator_name,
                self.specific_args.evaluators_cfg_path,
                process_id=i * group_n + j,
                group_id=j,
                env_id=i,
                start_message_list=None,
                single_chain_max_step=self.specific_args.single_chain_max_step,
                evaluation_times=self.specific_args.evaluation_times,
                tokenizer=self.tokenizer,
                eval_lock=self.eval_lock,
                specific_args=self.specific_args,
            )
            for i in range(env_num)
            for j in range(group_n)
        ]

    def select_random(self):
        self.envs_selected = random.sample(self._query_list, self.env_num)

    def select_sequential(self):

        if self.now_index >= len(self._query_list):
            self.now_index = 0
            print(f"\033[93m DataSet reset, reset the index to {self.now_index}\033[0m")

        self.envs_selected = self._query_list[self.now_index : self.now_index + self.env_num]

        self.now_index += len(self.envs_selected)

    def reset(self):
        self.envs_selected = []
        self.now_step = 0
        if self._is_random:
            self.select_random()
        else:
            self.select_sequential()
        self.rapidapi_wrappers = []
        observations = []
        infos = []
        for i, data_dict in enumerate(self.envs_selected):
            # query_list 已在 build_single_chain_envs 中按 white_list 过滤，此处仅计算 tool_des
            if "api_list" in data_dict:
                origin_tool_names = [standardize(cont["tool_name"]) for cont in data_dict["api_list"]]
                tool_des = contain(origin_tool_names, self._white_list)
                # if tool_des is False:
                #    continue
                tool_des = [[cont["standard_tool_name"], cont["description"]] for cont in tool_des]
            else:
                tool_des = None
            for j in range(self.group_n):
                idx = i * self.group_n + j
                self.rapidapi_wrappers.append(
                    rapidapi_wrapper(
                        data_dict,
                        tool_des,
                        self.retriever,
                        self.specific_args,
                        process_id=idx,
                    )
                )
                observation, info = self.envs[idx].reset(self.rapidapi_wrappers[idx], data_dict["query_id"])
                observations.append(observation)
                infos.append(info)

        return observations, infos

    def step(self, actions):

        async def run_async_steps():
            tasks = [self.envs[i].step(actions[i]) for i in range(len(self.envs))]
            return await asyncio.gather(*tasks)

        results = self.loop.run_until_complete(run_async_steps())
        self.now_step += 1

        print(f"\033[91mStep:{self.now_step}\033[0m")

        observations = [result[0] for result in results]
        scores = [result[1] for result in results]
        dones = [result[2] for result in results]
        infos = [result[3] for result in results]
        return observations, scores, dones, infos
