source /home/johnson/qiuyu/susu-xft/3rdparty/oneccl/build/_install/env/setvars.sh
export PYTHONPATH="${PYTHONPATH}:/home/johnson/qiuyu/susu-xft/src"
export $(python -c 'import xfastertransformer as xft; print(xft.get_env())')
TOKEN_PATH=/media/nvme/johnson/model-space/Llama-2-7b-hf
MODEL_PATH=/media/nvme/johnson/model-space/Llama-2-7b-xft
rm *.csv

OMP_NUM_THREADS=28 XFT_PIPELINE_STAGE=4  mpirun \
  -n 1 numactl --all -C 0-27 -m 0 \
    python -m vllm.entrypoints.openai.api_server \
      --model ${MODEL_PATH} \
      --tokenizer ${TOKEN_PATH} \
      --dtype bf16 \
      --kv-cache-dtype fp16 \
      --served-model-name xft \
      --port 8000 \
      --trust-remote-code \
  : -n 1 numactl --all -C 28-55 -m 0 \
    python -m vllm.entrypoints.slave \
      --dtype bf16 \
      --model ${MODEL_PATH} \
      --kv-cache-dtype fp16 \
  : -n 1 numactl --all -C 56-83 -m 1 \
    python -m vllm.entrypoints.slave \
      --dtype bf16 \
      --model ${MODEL_PATH} \
      --kv-cache-dtype fp16 \
  : -n 1 numactl --all -C 84-111 -m 1 \
    python -m vllm.entrypoints.slave \
      --dtype bf16 \
      --model ${MODEL_PATH} \
      --kv-cache-dtype fp16 \

# OMP_NUM_THREADS=28 XFT_PIPELINE_STAGE=4  mpirun \
#   -n 1 numactl --all -C 0-27 -m 0 \
#     python -m vllm.entrypoints.openai.api_server \
#       --model ${MODEL_PATH} \
#       --tokenizer ${TOKEN_PATH} \
#       --dtype bf16 \
#       --kv-cache-dtype fp16 \
#       --served-model-name xft \
#       --port 8000 \
#       --trust-remote-code \
#   : -n 1 numactl --all -C 28-55 -m 0 \
#     python -m vllm.entrypoints.slave \
#       --dtype bf16 \
#       --model ${MODEL_PATH} \
#       --kv-cache-dtype fp16 \
#   : -n 1 numactl --all -C 56-83 -m 1 \
#     python -m vllm.entrypoints.slave \
#       --dtype bf16 \
#       --model ${MODEL_PATH} \
#       --kv-cache-dtype fp16 \
#   : -n 1 numactl --all -C 84-111 -m 1 \
#     python -m vllm.entrypoints.slave \
#       --dtype bf16 \
#       --model ${MODEL_PATH} \
#       --kv-cache-dtype fp16 \