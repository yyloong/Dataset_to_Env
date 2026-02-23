# tmux 1:
cd /home/qianyy/Codes/StableToolBench
conda activate /home/qianyy/anaconda3/envs/dkv_vllm
vllm serve ./models/MirrorAPI-Cache --api-key EMPTY --port 12345 --served-model-name MirrorAPI-Cache

# tmux 2:
cd /home/qianyy/Codes/StableToolBench/server
python main_mirrorapi_cache.py