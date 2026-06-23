#!/bin/bash

set -e

qwen3_caption_model_path="${QWEN3_CAPTION_MODEL_PATH:-/home/tione/notebook/model/qwen3-omni}"
qwen3_30_model_path="${QWEN3_INSTRUCT_MODEL_PATH:-/home/tione/notebook/model/qwen3-30B}"

VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-80}
CAPTIONER_MAX_NUM_SEQS=${CAPTIONER_MAX_NUM_SEQS:-$VLLM_MAX_NUM_SEQS}
TEXT_MAX_NUM_SEQS=${TEXT_MAX_NUM_SEQS:-160}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-8192}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.95}
VLLM_COMPILATION_MODE=${VLLM_COMPILATION_MODE:-}
VLLM_ENFORCE_EAGER=${VLLM_ENFORCE_EAGER:-0}
VLLM_USE_TRITON_FLASH_ATTN=${VLLM_USE_TRITON_FLASH_ATTN:-0}
VLLM_ROCM_USE_AITER=${VLLM_ROCM_USE_AITER:-1}
VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-${ATTENTION_BACKEND:-ROCM_ATTN}}

export VLLM_USE_TRITON_FLASH_ATTN
export VLLM_ROCM_USE_AITER

qwen_vllm_extra_args=(--attention-backend "$VLLM_ATTENTION_BACKEND")
if [ -n "$VLLM_COMPILATION_MODE" ]; then
    qwen_vllm_extra_args+=(--compilation-config "{\"mode\": ${VLLM_COMPILATION_MODE}}")
fi
if [ "$VLLM_ENFORCE_EAGER" = "1" ]; then
    qwen_vllm_extra_args+=(--enforce-eager)
fi

if [ "$1" = "captioner_multi" ]; then
    NUM_INSTANCES=${2:-1}
    START_PORT=${CAPTION_BASE_PORT:-8901}
    GPU_OFFSET=${GPU_OFFSET:-0}

    echo "Starting ${NUM_INSTANCES} Qwen3-Omni captioner instance(s)..."

    for ((i=0; i<NUM_INSTANCES; i++)); do
        PORT=$((START_PORT + i))
        GPU_ID=$((GPU_OFFSET + i))

        echo "Starting captioner instance ${i} on GPU ${GPU_ID}, port ${PORT}..."
        VLLM_USE_TRITON_FLASH_ATTN=$VLLM_USE_TRITON_FLASH_ATTN \
        VLLM_USE_V1=0 \
        CUDA_VISIBLE_DEVICES=$GPU_ID \
        nohup vllm serve "$qwen3_caption_model_path" \
            --port "$PORT" \
            --host 127.0.0.1 \
            --served-model-name Qwen/Qwen3-Omni-30B-A3B-Captioner \
            --dtype bfloat16 \
            --max-model-len "$VLLM_MAX_MODEL_LEN" \
            --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
            --allowed-local-media-path / \
            --max-num-seqs "$CAPTIONER_MAX_NUM_SEQS" \
            --tensor-parallel-size 1 \
            --trust-remote-code \
            "${qwen_vllm_extra_args[@]}" > "vllm_captioner_${PORT}.log" 2>&1 &

        sleep 5
    done
    exit 0
fi

if [ "$1" = "instruct_multi" ]; then
    NUM_INSTANCES=${2:-1}
    START_PORT=${INSTRUCT_BASE_PORT:-9001}
    GPU_OFFSET=${GPU_OFFSET:-0}

    echo "Starting ${NUM_INSTANCES} Qwen3 instruct instance(s)..."

    for ((i=0; i<NUM_INSTANCES; i++)); do
        PORT=$((START_PORT + i))
        GPU_ID=$((GPU_OFFSET + i))

        echo "Starting instruct instance ${i} on GPU ${GPU_ID}, port ${PORT}..."
        VLLM_USE_TRITON_FLASH_ATTN=$VLLM_USE_TRITON_FLASH_ATTN \
        VLLM_USE_V1=0 \
        CUDA_VISIBLE_DEVICES=$GPU_ID \
        nohup vllm serve "$qwen3_30_model_path" \
            --port "$PORT" \
            --host 127.0.0.1 \
            --served-model-name Qwen/Qwen3-30B-A3B-Instruct-2507 \
            --dtype bfloat16 \
            --max-model-len "$VLLM_MAX_MODEL_LEN" \
            --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
            --max-num-seqs "$TEXT_MAX_NUM_SEQS" \
            --tensor-parallel-size 1 \
            --trust-remote-code \
            "${qwen_vllm_extra_args[@]}" > "vllm_instruct_${PORT}.log" 2>&1 &

        sleep 5
    done
    exit 0
fi

echo "Usage: $0 {captioner_multi|instruct_multi} [num_instances]" >&2
exit 1
