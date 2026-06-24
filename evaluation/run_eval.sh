#!/bin/bash
# run_eval.sh
# One-click evaluation pipeline for S2ST model outputs.
# Input: benchmark JSONL + results JSONL (per formats.md)
# Output: eval_results_{model_name}.jsonl + eval_summary_{model_name}.json

set -e

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# --- Input ---
BENCHMARK_FILE="${BENCHMARK_FILE:-}"
RESULTS_FILE="${RESULTS_FILE:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"                # Output directory
SPLIT="${SPLIT:-normal}"
SRC_LANG="${SRC_LANG:-zh}"
TGT_LANG="${TGT_LANG:-en}"

# --- Pipeline Module Paths ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EVAL_DIR="${SCRIPT_DIR}/eval"
CORE_MODULES="$(cd "${SCRIPT_DIR}/.." && pwd)/core_functional_modules"
export PYTHONPATH="${SCRIPT_DIR}/..:${PYTHONPATH:-}"
source "${SCRIPT_DIR}/service_orchestrator.sh"

# --- Service URLs / Service Orchestration ---
ASR_SERVER_URLS="${ASR_SERVER_URLS:-}"           # Reserved for compatibility; ASR runs as local worker processes
CAPTION_SERVER_URLS="${CAPTION_SERVER_URLS:-}"       # Used when AUTO_START_CAPTION_SERVERS=0
INSTRUCT_SERVER_URLS="${INSTRUCT_SERVER_URLS:-}"      # Used when AUTO_START_INSTRUCT_SERVERS=0
SERVER_HOST="${SERVER_HOST:-localhost}"
CAPTION_GPU_COUNT="${CAPTION_GPU_COUNT:-}"          # One captioner instance per GPU: ports 8901+
INSTRUCT_GPU_COUNT="${INSTRUCT_GPU_COUNT:-}"         # One instruct instance per GPU: ports 9001+
AUTO_START_CAPTION_SERVERS="${AUTO_START_CAPTION_SERVERS:-1}"
AUTO_START_INSTRUCT_SERVERS="${AUTO_START_INSTRUCT_SERVERS:-1}"
LLM_MODEL="${LLM_MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"

# --- Hardware ---
GPU_COUNT="${GPU_COUNT:-2}"
CAPTION_GPU_COUNT="${CAPTION_GPU_COUNT:-$GPU_COUNT}"
INSTRUCT_GPU_COUNT="${INSTRUCT_GPU_COUNT:-$GPU_COUNT}"
ASR_BATCH_SIZE="${ASR_BATCH_SIZE:-64}"
CAPTION_CONCURRENCY="${CAPTION_CONCURRENCY:-40}"
SED_BATCH_SIZE="${SED_BATCH_SIZE:-2048}"

# --- ASR Model Paths ---
ASR_MODEL_PATH="${ASR_MODEL_PATH:-}"
ALIGNER_MODEL_PATH="${ALIGNER_MODEL_PATH:-}"
ASR_VENV="${ASR_VENV:-}"                  # Path to ASR virtualenv (optional)
ASR_PYTHON="${ASR_PYTHON:-python}"        # Python executable used after ASR_VENV activation
ASR_ENFORCE_EAGER="${ASR_ENFORCE_EAGER:-0}"
ASR_VLLM_ROCM_USE_AITER="${ASR_VLLM_ROCM_USE_AITER:-${VLLM_ROCM_USE_AITER:-}}"

# --- Model Paths ---
BASE_COMET_MODEL="${BASE_COMET_MODEL:-}"
XCOMET_MODEL="${XCOMET_MODEL:-}"
COMET_QE_MODEL="${COMET_QE_MODEL:-}"            # Optional QE model for base COMET; leave empty to compute ref-only

# --- Feature Flags ---
ENABLE_SPEAKER_SIM="${ENABLE_SPEAKER_SIM:-}"        # Set to "--enable_speaker_sim" to enable
SPEAKER_SIM_PYTHON="${SPEAKER_SIM_PYTHON:-}"         # Isolated speaker-sim env python; see README uv setup
SPEAKER_SIM_CKPT="${SPEAKER_SIM_CKPT:-}"             # Path to wavlm_large_finetune.pth
SPEAKER_SIM_PROCS_PER_GPU="${SPEAKER_SIM_PROCS_PER_GPU:-2}"  # Processes per GPU (default 2)
ENABLE_LLM="${ENABLE_LLM:-}"                # Set to "--enable_llm" to enable
ENABLE_COMET="${ENABLE_COMET:-}"              # Set to "--enable_comet" to enable
ENABLE_XCOMET="${ENABLE_XCOMET:-}"            # Set to "--enable_xcomet" to enable
ASR_BASIC_ONLY="${ASR_BASIC_ONLY:-0}"
DISABLE_BASIC_AUDIO="${DISABLE_BASIC_AUDIO:-0}"
SLC_THRESHOLDS="${SLC_THRESHOLDS:-0.2,0.4}"
LLM_CONCURRENCY="${LLM_CONCURRENCY:-100}"
TEXT_CLIENT_CONCURRENCY="${TEXT_CLIENT_CONCURRENCY:-$LLM_CONCURRENCY}"
LLM_ENSEMBLE_RUNS="${LLM_ENSEMBLE_RUNS:-3}"
LLM_ENSEMBLE_STRATEGY="${LLM_ENSEMBLE_STRATEGY:-robust}"
LLM_PROMPT_VERSION="${LLM_PROMPT_VERSION:-default}"
EXTERNAL_HTTP_PROXY="${EXTERNAL_HTTP_PROXY:-}"
EXTERNAL_HTTPS_PROXY="${EXTERNAL_HTTPS_PROXY:-}"
LOCAL_NO_PROXY="${LOCAL_NO_PROXY:-127.0.0.1,localhost}"

# --- Phase Control ---
START_PHASE="${START_PHASE:-1}"                # 1=Data Prep, 2=Audio Extract, 3=Scoring
END_PHASE="${END_PHASE:-3}"
ISOLATE_PHASE3_METRIC_GROUPS="${ISOLATE_PHASE3_METRIC_GROUPS:-1}"
PHASE3_SHARDED_GROUPS="${PHASE3_SHARDED_GROUPS:-speaker_sim comet xcomet}"
SKIP_PHASE2_CAPTION="${SKIP_PHASE2_CAPTION:-0}"
SKIP_PHASE2_SUMMARY="${SKIP_PHASE2_SUMMARY:-0}"
SKIP_PHASE2_ASR="${SKIP_PHASE2_ASR:-0}"
SKIP_PHASE2_SED="${SKIP_PHASE2_SED:-0}"
SKIP_PHASE2_EVENT_COMBINE="${SKIP_PHASE2_EVENT_COMBINE:-0}"
MERGE_CAPTION_DIR="${MERGE_CAPTION_DIR:-}"
MERGE_SUMMARY_DIR="${MERGE_SUMMARY_DIR:-}"

# ==============================================================================
# INPUT VALIDATION
# ==============================================================================

trap cleanup_owned_services EXIT INT TERM

validate_positive_int() {
    local value="${1:?value is required}"
    local name="${2:?name is required}"

    if ! [[ "$value" =~ ^[0-9]+$ ]] || [ "$value" -le 0 ]; then
        echo "ERROR: ${name} must be a positive integer." >&2
        exit 1
    fi
}

wait_for_background_jobs() {
    local failed=0
    local status=0
    local pid

    for pid in "$@"; do
        if wait "$pid"; then
            continue
        else
            status=$?
            failed=1
            echo "ERROR: Background job failed (pid=${pid}, status=${status})." >&2
        fi
    done

    if [ "$failed" -ne 0 ]; then
        exit 1
    fi
}

group_uses_shards() {
    local group="${1:?group required}"
    local token

    if [ "${GPU_COUNT}" -le 1 ]; then
        return 1
    fi

    for token in ${PHASE3_SHARDED_GROUPS}; do
        if [ "${token}" = "${group}" ]; then
            return 0
        fi
    done

    return 1
}

disable_external_network() {
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
    export NO_PROXY="${LOCAL_NO_PROXY}"
}

enable_external_network() {
    export http_proxy="${EXTERNAL_HTTP_PROXY}"
    export https_proxy="${EXTERNAL_HTTPS_PROXY}"
    export HTTP_PROXY="${EXTERNAL_HTTP_PROXY}"
    export HTTPS_PROXY="${EXTERNAL_HTTPS_PROXY}"
    unset ALL_PROXY all_proxy
    export NO_PROXY="${LOCAL_NO_PROXY}"
}

run_speaker_sim_python() {
    local gpu_id="${1:?gpu id required}"
    shift

    env -u HIP_VISIBLE_DEVICES -u ROCR_VISIBLE_DEVICES \
        http_proxy="${EXTERNAL_HTTP_PROXY}" \
        https_proxy="${EXTERNAL_HTTPS_PROXY}" \
        HTTP_PROXY="${EXTERNAL_HTTP_PROXY}" \
        HTTPS_PROXY="${EXTERNAL_HTTPS_PROXY}" \
        NO_PROXY="${LOCAL_NO_PROXY}" \
        CUDA_VISIBLE_DEVICES="${gpu_id}" \
        "$SPEAKER_SIM_PYTHON" "$@"
}

if [ -z "$BENCHMARK_FILE" ] || [ -z "$RESULTS_FILE" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "ERROR: BENCHMARK_FILE, RESULTS_FILE, and OUTPUT_DIR must be set."
    exit 1
fi

PHASE2_ENABLED=0
PHASE3_ENABLED=0
PHASE3_LLM_ENABLED=0
USE_PHASE3_GROUP_RUNNER=0
RUN_EVENT_EVAL=1
DISABLE_LLM_EVENT_SCORER=""
DISABLE_LLM_EMOTION_SCORER="${DISABLE_LLM_EMOTION_SCORER:-}"
DISABLE_LLM_STYLE_SCORER="${DISABLE_LLM_STYLE_SCORER:-}"

if [ "${SPLIT}" = "normal" ]; then
    RUN_EVENT_EVAL=0
    DISABLE_LLM_EVENT_SCORER="--disable_llm_event"
else
    DISABLE_LLM_EMOTION_SCORER="${DISABLE_LLM_EMOTION_SCORER:---disable_llm_emotion}"
    DISABLE_LLM_STYLE_SCORER="${DISABLE_LLM_STYLE_SCORER:---disable_llm_style}"
fi

if [ -n "$DISABLE_LLM_EMOTION_SCORER" ] && [ "$DISABLE_LLM_EMOTION_SCORER" != "--disable_llm_emotion" ]; then
    DISABLE_LLM_EMOTION_SCORER="--disable_llm_emotion"
fi

if [ -n "$DISABLE_LLM_STYLE_SCORER" ] && [ "$DISABLE_LLM_STYLE_SCORER" != "--disable_llm_style" ]; then
    DISABLE_LLM_STYLE_SCORER="--disable_llm_style"
fi

if [[ "$START_PHASE" -le 2 ]] && [[ "$END_PHASE" -ge 2 ]]; then
    PHASE2_ENABLED=1
fi

if [[ "$START_PHASE" -le 3 ]] && [[ "$END_PHASE" -ge 3 ]]; then
    PHASE3_ENABLED=1
fi

if [[ "$PHASE3_ENABLED" -eq 1 ]] && [ -n "$ENABLE_LLM" ]; then
    PHASE3_LLM_ENABLED=1
fi

if [[ "$PHASE3_ENABLED" -eq 1 ]] && [ "$ISOLATE_PHASE3_METRIC_GROUPS" = "1" ]; then
    USE_PHASE3_GROUP_RUNNER=1
fi

if [ "$AUTO_START_CAPTION_SERVERS" = "1" ]; then
    validate_positive_int "$CAPTION_GPU_COUNT" "CAPTION_GPU_COUNT"
fi

if [ "$AUTO_START_INSTRUCT_SERVERS" = "1" ]; then
    validate_positive_int "$INSTRUCT_GPU_COUNT" "INSTRUCT_GPU_COUNT"
fi

if [ "$PHASE2_ENABLED" -eq 1 ] && [ "$SKIP_PHASE2_ASR" != "1" ]; then
    if [ -z "$ASR_MODEL_PATH" ] || [ -z "$ALIGNER_MODEL_PATH" ]; then
        echo "ERROR: ASR_MODEL_PATH and ALIGNER_MODEL_PATH must be set when Phase 2 ASR runs." >&2
        echo "       Set both variables, or use SKIP_PHASE2_ASR=1 if ASR outputs already exist." >&2
        exit 1
    fi
fi

if [ -n "$ENABLE_SPEAKER_SIM" ]; then
    if [ -z "$SPEAKER_SIM_PYTHON" ] || [ ! -f "$SPEAKER_SIM_PYTHON" ]; then
        echo "ERROR: SPEAKER_SIM_PYTHON must be set to the isolated env python." >&2
        echo "       Create the speaker-sim environment with uv as documented in README.md." >&2
        echo "       Then set SPEAKER_SIM_PYTHON to that environment's python." >&2
        exit 1
    fi
    if [ -z "$SPEAKER_SIM_CKPT" ] || [ ! -f "$SPEAKER_SIM_CKPT" ]; then
        echo "ERROR: SPEAKER_SIM_CKPT must point to wavlm_large_finetune.pth." >&2
        echo "       Download from: https://drive.google.com/file/d/1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP/view" >&2
        echo "       Set SPEAKER_SIM_CKPT to the downloaded wavlm_large_finetune.pth." >&2
        exit 1
    fi
    export SPEAKER_SIM_PYTHON SPEAKER_SIM_CKPT
fi

if [ "$PHASE2_ENABLED" -eq 1 ] && [ "$SKIP_PHASE2_CAPTION" != "1" ] && [ "$AUTO_START_CAPTION_SERVERS" != "1" ] && [ -z "$CAPTION_SERVER_URLS" ]; then
    echo "ERROR: CAPTION_SERVER_URLS must be set when AUTO_START_CAPTION_SERVERS=0 and Phase 2 runs." >&2
    exit 1
fi

if [ "$PHASE2_ENABLED" -eq 1 ] && [ "$SKIP_PHASE2_SUMMARY" != "1" ] && [ "$AUTO_START_INSTRUCT_SERVERS" != "1" ] && [ -z "$INSTRUCT_SERVER_URLS" ]; then
    echo "ERROR: INSTRUCT_SERVER_URLS must be set when AUTO_START_INSTRUCT_SERVERS=0 and Phase 2 Step 3 runs." >&2
    exit 1
fi

if [ "$PHASE3_LLM_ENABLED" -eq 1 ] && [ "$AUTO_START_INSTRUCT_SERVERS" != "1" ] && [ -z "$INSTRUCT_SERVER_URLS" ]; then
    echo "ERROR: INSTRUCT_SERVER_URLS must be set when AUTO_START_INSTRUCT_SERVERS=0 and ENABLE_LLM is set." >&2
    exit 1
fi

# ==============================================================================
# DERIVED PATHS
# ==============================================================================

EVAL_RECORDS="${OUTPUT_DIR}/eval_records.jsonl"
EVAL_RECORDS_MERGED="${OUTPUT_DIR}/eval_records_merged.jsonl"
HYP_PARQUET_DIR="${OUTPUT_DIR}/hyp_parquet"
HYP_TIMESTAMP_DIR="${OUTPUT_DIR}/hyp_timestamp"
HYP_CAPTION_DIR="${OUTPUT_DIR}/hyp_caption"
HYP_SUMMARY_DIR="${OUTPUT_DIR}/hyp_summary"
HYP_SED_DIR="${OUTPUT_DIR}/hyp_sed"
HYP_EVENTS_DIR="${OUTPUT_DIR}/hyp_events"
LOG_DIR="${OUTPUT_DIR}/logs"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

echo "=================================================="
echo "Evaluation Pipeline"
echo "Benchmark: $BENCHMARK_FILE"
echo "Results:   $RESULTS_FILE"
echo "Output:    $OUTPUT_DIR"
echo "Split:     $SPLIT"
echo "Phases:    $START_PHASE - $END_PHASE"
echo "GPUs:      $GPU_COUNT"
echo "Caption service mode:  $([ "$AUTO_START_CAPTION_SERVERS" = "1" ] && echo "auto-start (${CAPTION_GPU_COUNT} GPU(s))" || echo "manual")"
echo "Instruct service mode: $([ "$AUTO_START_INSTRUCT_SERVERS" = "1" ] && echo "auto-start (${INSTRUCT_GPU_COUNT} GPU(s))" || echo "manual")"
echo "=================================================="

disable_external_network

# ==============================================================================
# PHASE 1: Data Preparation
# ==============================================================================
if [[ "$START_PHASE" -le 1 ]] && [[ "$END_PHASE" -ge 1 ]]; then
    echo ">>> [Phase 1] Data Preparation (join benchmark + results)..."
    python -u "$EVAL_DIR/data_loader.py" \
        --benchmark "$BENCHMARK_FILE" \
        --results "$RESULTS_FILE" \
        --src_lang "$SRC_LANG" --tgt_lang "$TGT_LANG" \
        --output "$EVAL_RECORDS" \
        > "$LOG_DIR/phase1_data_loader.log" 2>&1
    echo "Phase 1 done."
fi

# ==============================================================================
# PHASE 2: Audio Feature Extraction (reuses core_functional_modules)
# ==============================================================================
if [[ "$START_PHASE" -le 2 ]] && [[ "$END_PHASE" -ge 2 ]]; then
    echo ">>> [Phase 2] Audio Feature Extraction..."

    # Step 0: Pack WAV → Parquet
    echo "  [Step 0] Pack WAV → Parquet..."
    python -u "$EVAL_DIR/prepare_hyp_parquet.py" \
        --input_jsonl "$EVAL_RECORDS" \
        --output_dir "$HYP_PARQUET_DIR" \
        > "$LOG_DIR/phase2_step0_parquet.log" 2>&1

    # Step 1: ASR + Timestamp (multi-GPU)
    if [ "$SKIP_PHASE2_ASR" = "1" ]; then
        echo "  [Step 1] Skipping ASR + Timestamp."
    else
        echo "  [Step 1] ASR + Timestamp ($GPU_COUNT GPUs)..."
        mkdir -p "$HYP_TIMESTAMP_DIR"

        # Clean stale locks (unconditional)
        python -c "
import sys; sys.path.insert(0, '$CORE_MODULES')
from utils.file_task_manager import cleanup_locks
cleanup_locks('$HYP_TIMESTAMP_DIR')
" 2>/dev/null || true

        asr_pids=()
        for ((i=0; i<GPU_COUNT; i++)); do
            (
                if [ -n "$ASR_VENV" ]; then
                    source "$ASR_VENV/bin/activate"
                fi
                CUDA_VISIBLE_DEVICES=$i \
                VLLM_ROCM_USE_AITER="${ASR_VLLM_ROCM_USE_AITER}" \
                "$ASR_PYTHON" -u "$CORE_MODULES/extract_timestamp/process_qwen3_asr.py" \
                    --input_dir "$HYP_PARQUET_DIR" \
                    --output_dir "$HYP_TIMESTAMP_DIR" \
                    --batch_size "$ASR_BATCH_SIZE" \
                    --file_suffix ".base.parquet" \
                    --asr_model_path "$ASR_MODEL_PATH" \
                    --aligner_model_path "$ALIGNER_MODEL_PATH" \
                    --enforce_eager "$ASR_ENFORCE_EAGER" \
                    --skip_meta
            ) > "$LOG_DIR/phase2_step1_asr_gpu${i}.log" 2>&1 &
            asr_pids+=($!)
            sleep 2
        done
        wait_for_background_jobs "${asr_pids[@]}"
    fi

    # Step 2: Caption (Qwen3-Omni-Captioner via VLLM HTTP)
    if [ "$SKIP_PHASE2_CAPTION" = "1" ]; then
        echo "  [Step 2] Skipping Audio Caption."
    else
        echo "  [Step 2] Audio Caption..."
        mkdir -p "$HYP_CAPTION_DIR"
        if [ "$AUTO_START_CAPTION_SERVERS" = "1" ]; then
            echo "  [Step 2] Starting caption VLLM services..."
            start_caption_servers "$CAPTION_GPU_COUNT" "$SERVER_HOST" >/dev/null
            CAPTION_SERVER_URLS="$(get_server_urls "$CAPTION_BASE_PORT" "$CAPTION_GPU_COUNT" "$SERVER_HOST")"
        fi
        python -u "$CORE_MODULES/captioner/qwen3_caption_server.py" \
            -i "$HYP_PARQUET_DIR" \
            -o "$HYP_CAPTION_DIR" \
            --server_url "$CAPTION_SERVER_URLS" \
            --concurrency "$CAPTION_CONCURRENCY" \
            > "$LOG_DIR/phase2_step2_caption.log" 2>&1
        if [ "$AUTO_START_CAPTION_SERVERS" = "1" ]; then
            stop_caption_servers
        fi
    fi

    # Step 3: Emotion/Style Summary (via Instruct VLLM)
    if [ "$SKIP_PHASE2_SUMMARY" = "1" ]; then
        echo "  [Step 3] Skipping Emotion/Style Summary."
    else
        echo "  [Step 3] Emotion/Style Summary..."
        mkdir -p "$HYP_SUMMARY_DIR"
        if [ "$AUTO_START_INSTRUCT_SERVERS" = "1" ] && [ "${INSTRUCT_SERVERS_OWNED:-0}" != "1" ]; then
            echo "  [Step 3] Starting instruct VLLM services..."
            start_instruct_servers "$INSTRUCT_GPU_COUNT" "$SERVER_HOST" >/dev/null
            INSTRUCT_SERVER_URLS="$(get_server_urls "$INSTRUCT_BASE_PORT" "$INSTRUCT_GPU_COUNT" "$SERVER_HOST")"
        fi
        python -u "$CORE_MODULES/captioner/emotion_style_summary.py" \
            -i "$HYP_CAPTION_DIR" \
            -o "$HYP_SUMMARY_DIR" \
            --server_url "$INSTRUCT_SERVER_URLS" \
            --concurrency "$TEXT_CLIENT_CONCURRENCY" \
            > "$LOG_DIR/phase2_step3_summary.log" 2>&1
        if [ "$AUTO_START_INSTRUCT_SERVERS" = "1" ]; then
            stop_instruct_servers
        fi
    fi

    if [ "$RUN_EVENT_EVAL" = "1" ]; then
        # Step 4: SED (BEATs, multi-GPU)
        if [ "$SKIP_PHASE2_SED" = "1" ]; then
            echo "  [Step 4] Skipping Sound Event Detection."
        else
            echo "  [Step 4] Sound Event Detection ($GPU_COUNT GPUs)..."
            mkdir -p "$HYP_SED_DIR"
            sed_pids=()
            for ((i=0; i<GPU_COUNT; i++)); do
                CUDA_VISIBLE_DEVICES=$i python -u "$CORE_MODULES/PretrainedSED/batch_inference.py" \
                    --parquet_path "$HYP_PARQUET_DIR" \
                    --output_jsonl "$HYP_SED_DIR" \
                    --model_name "BEATs" \
                    --cuda \
                    --batch_size "$SED_BATCH_SIZE" \
                    > "$LOG_DIR/phase2_step4_sed_gpu${i}.log" 2>&1 &
                sed_pids+=($!)
                sleep 2
            done
            wait_for_background_jobs "${sed_pids[@]}"
        fi

        # Step 5: Event Combine (CPU)
        if [ "$SKIP_PHASE2_EVENT_COMBINE" = "1" ]; then
            echo "  [Step 5] Skipping Event Combine."
        else
            echo "  [Step 5] Event Combine..."
            mkdir -p "$HYP_EVENTS_DIR"
            python -u "$CORE_MODULES/extract_timestamp/combine_time_event.py" \
                --timestamp_file "$HYP_TIMESTAMP_DIR" \
                --event_file "$HYP_SED_DIR" \
                --output_file "$HYP_EVENTS_DIR" \
                --threshold 0.5 \
                > "$LOG_DIR/phase2_step5_event_combine.log" 2>&1
        fi
    else
        echo "  [Step 4-5] Skipping event extraction for SPLIT=${SPLIT}."
    fi

    # Step 6: Merge features back into eval records
    echo "  [Step 6] Merge features..."
    EVENT_DIR_ARGS=()
    if [ "$RUN_EVENT_EVAL" = "1" ]; then
        EVENT_DIR_ARGS=(--event_dir "$HYP_EVENTS_DIR")
    fi
    MERGE_CAPTION_DIR="${MERGE_CAPTION_DIR:-$HYP_CAPTION_DIR}"
    MERGE_SUMMARY_DIR="${MERGE_SUMMARY_DIR:-$HYP_SUMMARY_DIR}"
    python -u "$EVAL_DIR/merge_hyp_features.py" \
        --input_jsonl "$EVAL_RECORDS" \
        --asr_dir "$HYP_TIMESTAMP_DIR" \
        --caption_dir "$MERGE_CAPTION_DIR" \
        --summary_dir "$MERGE_SUMMARY_DIR" \
        "${EVENT_DIR_ARGS[@]}" \
        --output_jsonl "$EVAL_RECORDS_MERGED" \
        > "$LOG_DIR/phase2_step6_merge.log" 2>&1

    echo "Phase 2 done."
fi

# ==============================================================================
# PHASE 3: Scoring (all dimensions)
# ==============================================================================
if [[ "$START_PHASE" -le 3 ]] && [[ "$END_PHASE" -ge 3 ]]; then
    echo ">>> [Phase 3] Scoring all dimensions..."

    # Use merged if available, otherwise use raw eval_records
    INPUT_FOR_SCORING="$EVAL_RECORDS_MERGED"
    if [ ! -f "$INPUT_FOR_SCORING" ]; then
        INPUT_FOR_SCORING="$EVAL_RECORDS"
    fi

    COMET_ARGS=""
    if [ -n "$BASE_COMET_MODEL" ]; then
        COMET_ARGS="--base_comet_model $BASE_COMET_MODEL"
        if [ -n "$COMET_QE_MODEL" ]; then
            COMET_ARGS="$COMET_ARGS --comet_qe_model $COMET_QE_MODEL"
        fi
    fi

    XCOMET_ARGS=""
    if [ -n "$XCOMET_MODEL" ]; then
        XCOMET_ARGS="--xcomet_model $XCOMET_MODEL"
    fi

    LLM_ARGS=""
    if [ -n "$ENABLE_LLM" ] && [ "$AUTO_START_INSTRUCT_SERVERS" = "1" ] && [ "$USE_PHASE3_GROUP_RUNNER" != "1" ] && [ "${INSTRUCT_SERVERS_OWNED:-0}" != "1" ]; then
        echo "  [Phase 3] Starting instruct VLLM services for LLM scorers..."
        start_instruct_servers "$INSTRUCT_GPU_COUNT" "$SERVER_HOST" >/dev/null
        INSTRUCT_SERVER_URLS="$(get_server_urls "$INSTRUCT_BASE_PORT" "$INSTRUCT_GPU_COUNT" "$SERVER_HOST")"
    fi
    if [ -n "$INSTRUCT_SERVER_URLS" ] && [ -n "$ENABLE_LLM" ]; then
        LLM_ARGS="--llm_url $INSTRUCT_SERVER_URLS --llm_model $LLM_MODEL --llm_concurrency $LLM_CONCURRENCY --llm_ensemble_runs $LLM_ENSEMBLE_RUNS --llm_ensemble_strategy $LLM_ENSEMBLE_STRATEGY --llm_prompt_version $LLM_PROMPT_VERSION"
    fi

    PHASE3_ISOLATION_ARG=""
    if [ "$ISOLATE_PHASE3_METRIC_GROUPS" = "1" ]; then
        PHASE3_ISOLATION_ARG="--isolate_metric_groups"
    fi

    if [ "$USE_PHASE3_GROUP_RUNNER" = "1" ]; then
        PHASE3_GROUP_RESULTS_DIR="${OUTPUT_DIR}/phase3_group_results.$$"
        mkdir -p "$PHASE3_GROUP_RESULTS_DIR"
        : > "$LOG_DIR/phase3_scoring.log"

        EXISTING_PHASE3_GROUP_RESULTS_PATH=""
        EXISTING_PHASE3_GROUP_CORPUS_PATH=""
        EXISTING_PHASE3_SHARD_RESULTS_DIR=""

        resolve_existing_phase3_group_paths() {
            local group="${1:?group is required}"
            local candidate_dir
            local result_path
            local corpus_path

            EXISTING_PHASE3_GROUP_RESULTS_PATH=""
            EXISTING_PHASE3_GROUP_CORPUS_PATH=""
            EXISTING_PHASE3_SHARD_RESULTS_DIR=""

            local candidate_list
            candidate_list="$(mktemp "${OUTPUT_DIR}/phase3_candidates.XXXXXX")"
            find "$OUTPUT_DIR" -maxdepth 1 -mindepth 1 -type d -name 'phase3_group_results.*' ! -path "$PHASE3_GROUP_RESULTS_DIR" -printf '%T@ %p\n' \
                | sort -nr \
                | cut -d' ' -f2- \
                > "$candidate_list"

            while IFS= read -r candidate_dir; do
                [ -n "$candidate_dir" ] || continue
                result_path="$candidate_dir/${group}_results.json"
                corpus_path="$candidate_dir/${group}_corpus.json"
                if [ -f "$result_path" ]; then
                    EXISTING_PHASE3_GROUP_RESULTS_PATH="$result_path"
                fi
                if [ -f "$corpus_path" ]; then
                    EXISTING_PHASE3_GROUP_CORPUS_PATH="$corpus_path"
                fi
                if [ -z "$EXISTING_PHASE3_SHARD_RESULTS_DIR" ] && [ -d "$candidate_dir/${group}" ]; then
                    if find "$candidate_dir/${group}" -maxdepth 1 -type f -name 'shard_*.json' ! -name '*_corpus.json' -print -quit | grep -q .; then
                        EXISTING_PHASE3_SHARD_RESULTS_DIR="$candidate_dir/${group}"
                    fi
                fi
                if [ -n "$EXISTING_PHASE3_GROUP_RESULTS_PATH" ] && [ -n "$EXISTING_PHASE3_GROUP_CORPUS_PATH" ]; then
                    rm -f "$candidate_list"
                    return 0
                fi
            done < "$candidate_list"
            rm -f "$candidate_list"

            return 0
        }

        reuse_existing_phase3_group_if_complete() {
            local group="${1:?group is required}"
            resolve_existing_phase3_group_paths "$group"
            if [ -z "$EXISTING_PHASE3_GROUP_RESULTS_PATH" ] || [ -z "$EXISTING_PHASE3_GROUP_CORPUS_PATH" ]; then
                return 1
            fi

            cp "$EXISTING_PHASE3_GROUP_RESULTS_PATH" "$PHASE3_GROUP_RESULTS_DIR/${group}_results.json"
            cp "$EXISTING_PHASE3_GROUP_CORPUS_PATH" "$PHASE3_GROUP_RESULTS_DIR/${group}_corpus.json"
            echo "  [Phase 3] Reusing existing ${group} results from $(dirname "$EXISTING_PHASE3_GROUP_RESULTS_PATH")" >> "$LOG_DIR/phase3_scoring.log"
            return 0
        }

        run_phase3_group() {
            local group="${1:?group is required}"
            local group_llm_args="${2:-}"
            local shard_dir="$PHASE3_GROUP_RESULTS_DIR/${group}"
            local pids=()
            local rank
            local existing_group_results_args=()
            local asr_basic_args=()

            if reuse_existing_phase3_group_if_complete "$group"; then
                return 0
            fi

            if [ -n "$EXISTING_PHASE3_GROUP_RESULTS_PATH" ]; then
                existing_group_results_args=(--existing_group_results_path "$EXISTING_PHASE3_GROUP_RESULTS_PATH")
            fi
            if [ "$ASR_BASIC_ONLY" = "1" ]; then
                asr_basic_args+=(--asr_basic_only)
            fi
            if [ "$DISABLE_BASIC_AUDIO" = "1" ]; then
                asr_basic_args+=(--disable_basic_audio)
            fi

            if group_uses_shards "$group"; then
                mkdir -p "$shard_dir"
                if [ -n "$EXISTING_PHASE3_SHARD_RESULTS_DIR" ] && [ "$EXISTING_PHASE3_SHARD_RESULTS_DIR" != "$shard_dir" ]; then
                    local existing_idx=0
                    local existing_shard
                    while IFS= read -r existing_shard; do
                        cp "$existing_shard" "$shard_dir/shard_existing_${existing_idx}.json"
                        existing_idx=$((existing_idx + 1))
                    done < <(find "$EXISTING_PHASE3_SHARD_RESULTS_DIR" -maxdepth 1 -type f -name 'shard_*.json' ! -name '*_corpus.json' | sort)
                    if [ "$existing_idx" -gt 0 ]; then
                        echo "  [Phase 3] Seeded ${group} resume with ${existing_idx} existing shard file(s) from ${EXISTING_PHASE3_SHARD_RESULTS_DIR}" >> "$LOG_DIR/phase3_scoring.log"
                    fi
                fi

                # speaker_sim uses an isolated env and supports multiple procs per GPU.
                if [ "$group" = "speaker_sim" ]; then
                    local world_size=$(( GPU_COUNT * SPEAKER_SIM_PROCS_PER_GPU ))
                    for ((rank=0; rank<world_size; rank++)); do
                        local gpu_id=$(( rank / SPEAKER_SIM_PROCS_PER_GPU ))
                        (
                            run_speaker_sim_python "$gpu_id" -u "$EVAL_DIR/run_full_eval.py" \
                                --input "$INPUT_FOR_SCORING" \
                                --output_dir "$OUTPUT_DIR" \
                                --src_lang "$SRC_LANG" --tgt_lang "$TGT_LANG" \
                                --slc_thresholds "$SLC_THRESHOLDS" \
                                --score_group speaker_sim \
                                --partial_results_path "$shard_dir/shard_${rank}.json" \
                                --partial_corpus_path "$shard_dir/shard_${rank}_corpus.json" \
                                --phase3_rank "$rank" \
                                --phase3_world_size "$world_size" \
                                "${existing_group_results_args[@]}" \
                                --shard_results_dir "$shard_dir" \
                                --speaker_sim_ckpt "$SPEAKER_SIM_CKPT" \
                                --speaker_sim_python "$SPEAKER_SIM_PYTHON" \
                                $ENABLE_SPEAKER_SIM
                        ) >> "$LOG_DIR/phase3_scoring.log" 2>&1 &
                        pids+=($!)
                    done
                else
                    for ((rank=0; rank<GPU_COUNT; rank++)); do
                        (
                            CUDA_VISIBLE_DEVICES=$rank python -u "$EVAL_DIR/run_full_eval.py" \
                                --input "$INPUT_FOR_SCORING" \
                                --output_dir "$OUTPUT_DIR" \
                                --src_lang "$SRC_LANG" --tgt_lang "$TGT_LANG" \
                                --slc_thresholds "$SLC_THRESHOLDS" \
                                --score_group "$group" \
                                --partial_results_path "$shard_dir/shard_${rank}.json" \
                                --partial_corpus_path "$shard_dir/shard_${rank}_corpus.json" \
                                --phase3_rank "$rank" \
                                --phase3_world_size "$GPU_COUNT" \
                                "${existing_group_results_args[@]}" \
                                --shard_results_dir "$shard_dir" \
                                $ENABLE_SPEAKER_SIM \
                                $ENABLE_LLM \
                                $ENABLE_COMET \
                                $ENABLE_XCOMET \
                                $DISABLE_LLM_EMOTION_SCORER \
                                $DISABLE_LLM_STYLE_SCORER \
                                $DISABLE_LLM_EVENT_SCORER \
                                "${asr_basic_args[@]}" \
                                $COMET_ARGS \
                                $XCOMET_ARGS \
                                $group_llm_args
                        ) >> "$LOG_DIR/phase3_scoring.log" 2>&1 &
                        pids+=($!)
                    done
                fi
                wait_for_background_jobs "${pids[@]}"
            else
                python -u "$EVAL_DIR/run_full_eval.py" \
                    --input "$INPUT_FOR_SCORING" \
                    --output_dir "$OUTPUT_DIR" \
                    --src_lang "$SRC_LANG" --tgt_lang "$TGT_LANG" \
                    --slc_thresholds "$SLC_THRESHOLDS" \
                    --score_group "$group" \
                    --partial_results_path "$PHASE3_GROUP_RESULTS_DIR/${group}_results.json" \
                    --partial_corpus_path "$PHASE3_GROUP_RESULTS_DIR/${group}_corpus.json" \
                    $ENABLE_SPEAKER_SIM \
                    $ENABLE_LLM \
                    $ENABLE_COMET \
                    $ENABLE_XCOMET \
                    $DISABLE_LLM_EMOTION_SCORER \
                    $DISABLE_LLM_STYLE_SCORER \
                    $DISABLE_LLM_EVENT_SCORER \
                    "${asr_basic_args[@]}" \
                    $COMET_ARGS \
                    $XCOMET_ARGS \
                    $group_llm_args \
                    >> "$LOG_DIR/phase3_scoring.log" 2>&1
            fi
        }

        if [ "$DISABLE_BASIC_AUDIO" != "1" ]; then
            run_phase3_group "bleu_audio"
        fi

        if [ -n "$ENABLE_LLM" ]; then
            echo "  [Phase 3] Starting instruct VLLM services for isolated LLM group..." >> "$LOG_DIR/phase3_scoring.log"
            PHASE3_STARTED_INSTRUCT_SERVERS=0
            if [ "$AUTO_START_INSTRUCT_SERVERS" = "1" ]; then
                start_instruct_servers "$INSTRUCT_GPU_COUNT" "$SERVER_HOST" >> "$LOG_DIR/phase3_scoring.log" 2>&1
                INSTRUCT_SERVER_URLS="$(get_server_urls "$INSTRUCT_BASE_PORT" "$INSTRUCT_GPU_COUNT" "$SERVER_HOST")"
                PHASE3_STARTED_INSTRUCT_SERVERS=1
            fi
            LLM_ARGS="--llm_url $INSTRUCT_SERVER_URLS --llm_model $LLM_MODEL --llm_concurrency $LLM_CONCURRENCY --llm_ensemble_runs $LLM_ENSEMBLE_RUNS --llm_ensemble_strategy $LLM_ENSEMBLE_STRATEGY --llm_prompt_version $LLM_PROMPT_VERSION"
            run_phase3_group "llm" "$LLM_ARGS"
            if [ "$PHASE3_STARTED_INSTRUCT_SERVERS" = "1" ]; then
                stop_instruct_servers >> "$LOG_DIR/phase3_scoring.log" 2>&1
            fi
        fi

        if [ -n "$ENABLE_SPEAKER_SIM" ]; then
            run_phase3_group "speaker_sim"
        fi

        enable_external_network

        if [ -n "$ENABLE_COMET" ] && [ -n "$BASE_COMET_MODEL" ]; then
            run_phase3_group "comet"
        fi

        if [ -n "$ENABLE_XCOMET" ] && [ -n "$XCOMET_MODEL" ]; then
            run_phase3_group "xcomet"
        fi

        python -u "$EVAL_DIR/run_full_eval.py" \
            --input "$INPUT_FOR_SCORING" \
            --output_dir "$OUTPUT_DIR" \
            --src_lang "$SRC_LANG" --tgt_lang "$TGT_LANG" \
            --slc_thresholds "$SLC_THRESHOLDS" \
            --aggregate_group_results_dir "$PHASE3_GROUP_RESULTS_DIR" \
            $ENABLE_SPEAKER_SIM \
            $ENABLE_LLM \
            $ENABLE_COMET \
            $ENABLE_XCOMET \
            $DISABLE_LLM_EMOTION_SCORER \
            $DISABLE_LLM_STYLE_SCORER \
            $DISABLE_LLM_EVENT_SCORER \
            $([ "$ASR_BASIC_ONLY" = "1" ] && printf %s "--asr_basic_only") \
            $([ "$DISABLE_BASIC_AUDIO" = "1" ] && printf %s "--disable_basic_audio") \
            $COMET_ARGS \
            $XCOMET_ARGS \
            >> "$LOG_DIR/phase3_scoring.log" 2>&1
    else
        enable_external_network
        python -u "$EVAL_DIR/run_full_eval.py" \
            --input "$INPUT_FOR_SCORING" \
            --output_dir "$OUTPUT_DIR" \
            --src_lang "$SRC_LANG" --tgt_lang "$TGT_LANG" \
            --slc_thresholds "$SLC_THRESHOLDS" \
            $PHASE3_ISOLATION_ARG \
            $ENABLE_SPEAKER_SIM \
            $ENABLE_LLM \
            $ENABLE_COMET \
            $ENABLE_XCOMET \
            $DISABLE_LLM_EMOTION_SCORER \
            $DISABLE_LLM_STYLE_SCORER \
            $DISABLE_LLM_EVENT_SCORER \
            $([ "$ASR_BASIC_ONLY" = "1" ] && printf %s "--asr_basic_only") \
            $([ "$DISABLE_BASIC_AUDIO" = "1" ] && printf %s "--disable_basic_audio") \
            $COMET_ARGS \
            $XCOMET_ARGS \
            $LLM_ARGS \
            > "$LOG_DIR/phase3_scoring.log" 2>&1
    fi

    if [ -n "$ENABLE_LLM" ] && [ "$AUTO_START_INSTRUCT_SERVERS" = "1" ]; then
        stop_instruct_servers
    fi

    echo "Phase 3 done."
fi

disable_external_network

echo "=================================================="
echo "Evaluation Pipeline Complete!"
echo "Results in: $OUTPUT_DIR"
echo "=================================================="
