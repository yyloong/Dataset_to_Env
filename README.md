# Dataset_to_Env

将主流工具调用 / 函数调用评测数据集封装为 **RL 环境**，支持使用本地 vLLM 进行强化学习训练或离线评测。

## 功能概览

| 子项目 | 数据集 | 功能 |
|--------|--------|------|
| **StableToolBench** | ToolBench / StableToolBench | 多轮工具调用任务 → RL 环境 |
| **BFCL** | Berkeley Function Calling Leaderboard | 函数调用任务 → RL 环境 |

两者均提供 `step` / `reset` 接口，可与 RL 算法（如 PPO、DPO）对接，用于训练或评估工具/函数调用能力。

## 项目结构

```
Dataset_to_Env/
├── StableToolBench/           # ToolBench 环境
│   ├── toolbench_env/        # 环境构建与测试
│   │   ├── build_env.py      # 构建 RL 环境
│   │   └── test_env_tokenized.py   # 环境测试（本地 vLLM）
│   ├── convert.sh            # 答案格式转换
│   ├── eval.sh               # 评测脚本
│   └── inference.sh          # 推理脚本
│
└── berkeley-function-call-leaderboard/   # BFCL 环境
    ├── bfcl_env/             # 环境封装
    │   ├── bfcl_env.py       # 构建 BFCL RL 环境
    │   └── bfcl_env_test_with_same_calling.py   # 环境测试（本地 vLLM）
    └── environment.sh        # 依赖安装
```

## 快速开始

### 1. StableToolBench 环境

```bash
cd StableToolBench
# 1. 启动 vLLM 服务（端口 8000）
bash start_vllm.sh

# 2. 启动 API 模拟服务（MirrorAPI 或 Cache）
bash start_server.sh

# 3. 运行环境测试
cd toolbench_env && python test_env_tokenized.py
```

### 2. BFCL 环境

```bash
cd berkeley-function-call-leaderboard
# 1. 安装依赖
bash environment.sh

# 2. 启动 vLLM（端口 8000）

# 3. 运行环境测试
cd bfcl_env && python bfcl_env_test_with_same_calling.py
```

### 配置说明

- **vLLM 地址**：默认 `http://127.0.0.1:8000/v1`
- **模型**：在对应测试脚本中修改 `VLLM_MODEL_NAME`、`MODEL_PATH`
- **测试类别**：BFCL 支持 `live_simple`、`live_multiple`、`memory`、`web_search` 等

## 环境接口

两个环境均遵循标准 RL 接口：

- `reset()` → `observations`, `infos`
- `step(actions)` → `observations`, `rewards`, `dones`, `infos`

`observations` 为当前 prompt 文本，`actions` 为模型生成的工具调用或函数调用，`rewards` 由各自评测逻辑计算。

## 依赖

- Python 3.10+
- vLLM（本地模型推理）
- StableToolBench：见 `StableToolBench/requirements.txt`
- BFCL：见 `berkeley-function-call-leaderboard/environment.sh`

## 参考

- [StableToolBench](https://github.com/zhichengg/StableToolBench)
- [Berkeley Function Calling Leaderboard](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard)
