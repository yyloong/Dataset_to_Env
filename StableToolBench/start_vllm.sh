CUDA_VISIBLE_DEVICES=6 python -m vllm.entrypoints.openai.api_server \
    --model ToolBench/ToolLLaMA-2-7b-v2 \
    --dtype auto \
    --data-parallel-size 1 \
    --seed 42 \
    --port 8000 \
    --gpu-memory-utilization 0.8 \
    --tool-call-parser "openai" \
    #--served-model-name "Qwen/Qwen3-8B"
    #--enforce-eager \
    # --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    # --cpu-offload-gb $OFF_LOAD \
    #--enable-auto-tool-choice \
    #--host 127.0.0.1
    #--swap_space $SWAP_SPACE \