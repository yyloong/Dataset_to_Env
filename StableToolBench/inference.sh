export CUDA_VISIBLE_DEVICES=0,1
export TOOLBENCH_KEY="empty"
export SERVICE_URL="http://127.0.0.1:12001/virtual"
export OUTPUT_DIR="data/qwen_cot"
export no_proxy="localhost,127.0.0.1"
export PYTHONPATH=./
group=G1_instruction
mkdir -p $OUTPUT_DIR; mkdir -p $OUTPUT_DIR/$group

python toolbench/inference/qa_pipeline_multithread.py \
    --tool_root_dir tools_folder/server_cache/tools \
    --backbone_model toolllama_vllm \
    --model_path Qwen/Qwen3-8B \
    --max_observation_length 1024 \
    --observ_compress_method truncate \
    --base_url "http://127.0.0.1:8000/v1" \
    --method CoT@1 \
    --single_chain_max_step 12 \
    --input_query_file "solvable_queries/test_instruction/G1_instruction.json" \
    --output_answer_file $OUTPUT_DIR \
    --toolbench_key $TOOLBENCH_KEY \
    --num_thread 100