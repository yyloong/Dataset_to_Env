export CONVERTED_ANSWER_PATH=data_converted
export SAVE_PATH=data_pass_rate_results
export CANDIDATE_MODEL=qwen2.5-14B_cot

# 取消 OPENAI_KEY 环境变量，避免与 api_key.json 冲突
unset OPENAI_KEY

# 创建输出目录
mkdir -p ${SAVE_PATH}

# 启用调试模式（可选）
export DEBUG_API=1

python toolbench/tooleval/eval_pass_rate.py \
    --converted_answer_path ${CONVERTED_ANSWER_PATH} \
    --save_path ${SAVE_PATH} \
    --reference_model ${CANDIDATE_MODEL} \
    --test_ids solvable_queries/test_query_ids \
    --max_eval_threads 1 \
    --evaluate_times 1 \
    --test_set G1_instruction \
    --evaluator tooleval_deepseek-normalization       