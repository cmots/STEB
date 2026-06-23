#!/bin/bash

SERVICE_ORCHESTRATOR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_ORCHESTRATOR_ROOT="$(cd "${SERVICE_ORCHESTRATOR_DIR}/.." && pwd)"
SERVICE_ORCHESTRATOR_START_SCRIPT="${SERVICE_ORCHESTRATOR_START_SCRIPT:-${SERVICE_ORCHESTRATOR_ROOT}/core_functional_modules/start_servers.sh}"

CAPTION_BASE_PORT="${CAPTION_BASE_PORT:-8901}"
INSTRUCT_BASE_PORT="${INSTRUCT_BASE_PORT:-9001}"
MAX_SERVICE_PORT_SPAN="${MAX_SERVICE_PORT_SPAN:-32}"
SERVICE_READY_RETRIES="${SERVICE_READY_RETRIES:-180}"
SERVICE_READY_SLEEP_SECONDS="${SERVICE_READY_SLEEP_SECONDS:-2}"
SERVICE_STOP_RETRIES="${SERVICE_STOP_RETRIES:-120}"
SERVICE_STOP_SLEEP_SECONDS="${SERVICE_STOP_SLEEP_SECONDS:-2}"

: "${CAPTION_SERVERS_OWNED:=0}"
: "${INSTRUCT_SERVERS_OWNED:=0}"
: "${CAPTION_OWNED_PORTS:=}"
: "${INSTRUCT_OWNED_PORTS:=}"


get_server_urls() {
    local start_port="${1:?start_port is required}"
    local instance_count="${2:?instance_count is required}"
    local host="${3:-127.0.0.1}"
    local i
    local port
    local -a urls=()

    if ! [[ "$instance_count" =~ ^[0-9]+$ ]] || [ "$instance_count" -le 0 ]; then
        echo "instance_count must be a positive integer" >&2
        return 1
    fi

    for ((i = 0; i < instance_count; i++)); do
        port=$((start_port + i))
        urls+=("http://${host}:${port}/v1")
    done

    local IFS=,
    printf '%s\n' "${urls[*]}"
}


wait_for_http_ready() {
    local host="${1:?host is required}"
    local port="${2:?port is required}"
    local ready_url="http://${host}:${port}/v1/models"
    local attempt

    for ((attempt = 1; attempt <= SERVICE_READY_RETRIES; attempt++)); do
        if curl --noproxy '*' --silent --show-error --fail "$ready_url" >/dev/null 2>&1; then
            return 0
        fi
        sleep "$SERVICE_READY_SLEEP_SECONDS"
    done

    echo "Timed out waiting for ${ready_url}" >&2
    return 1
}


wait_for_http_stopped() {
    local host="${1:?host is required}"
    local port="${2:?port is required}"
    local ready_url="http://${host}:${port}/v1/models"
    local attempt

    for ((attempt = 1; attempt <= SERVICE_STOP_RETRIES; attempt++)); do
        if ! curl --noproxy '*' --silent --show-error --fail --max-time 2 "$ready_url" >/dev/null 2>&1; then
            return 0
        fi
        sleep "$SERVICE_STOP_SLEEP_SECONDS"
    done

    echo "Timed out waiting for ${ready_url} to stop" >&2
    return 1
}


_validate_instance_count() {
    local instance_count="${1:?instance_count is required}"

    if ! [[ "$instance_count" =~ ^[0-9]+$ ]] || [ "$instance_count" -le 0 ]; then
        echo "instance_count must be a positive integer" >&2
        return 1
    fi
}


_record_owned_ports() {
    local start_port="${1:?start_port is required}"
    local instance_count="${2:?instance_count is required}"
    local i
    local ports=""

    for ((i = 0; i < instance_count; i++)); do
        ports="${ports:+${ports} }$((start_port + i))"
    done

    printf '%s\n' "$ports"
}


_wait_for_owned_ports() {
    local host="${1:?host is required}"
    local ports="${2:-}"
    local port

    for port in $ports; do
        wait_for_http_ready "$host" "$port"
    done
}


_wait_for_stopped_ports() {
    local host="${1:?host is required}"
    local ports="${2:-}"
    local port

    for port in $ports; do
        wait_for_http_stopped "$host" "$port"
    done
}


start_caption_servers() {
    local instance_count="${1:?instance_count is required}"
    local host="${2:-127.0.0.1}"
    local launch_pythonpath="${SERVICE_ORCHESTRATOR_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

    _validate_instance_count "$instance_count" || return 1

    PYTHONPATH="$launch_pythonpath" bash "$SERVICE_ORCHESTRATOR_START_SCRIPT" captioner_multi "$instance_count" >&2

    CAPTION_OWNED_PORTS="$(_record_owned_ports "$CAPTION_BASE_PORT" "$instance_count")"
    CAPTION_SERVERS_OWNED=1
    _wait_for_owned_ports "$host" "$CAPTION_OWNED_PORTS" || return 1
    get_server_urls "$CAPTION_BASE_PORT" "$instance_count" "$host"
}


start_instruct_servers() {
    local instance_count="${1:?instance_count is required}"
    local host="${2:-127.0.0.1}"
    local launch_pythonpath="${SERVICE_ORCHESTRATOR_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

    _validate_instance_count "$instance_count" || return 1

    PYTHONPATH="$launch_pythonpath" bash "$SERVICE_ORCHESTRATOR_START_SCRIPT" instruct_multi "$instance_count" >&2

    INSTRUCT_OWNED_PORTS="$(_record_owned_ports "$INSTRUCT_BASE_PORT" "$instance_count")"
    INSTRUCT_SERVERS_OWNED=1
    _wait_for_owned_ports "$host" "$INSTRUCT_OWNED_PORTS" || return 1
    get_server_urls "$INSTRUCT_BASE_PORT" "$instance_count" "$host"
}


_kill_owned_ports() {
    local ports="${1:-}"
    local port
    local pids

    for port in $ports; do
        pids="$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
        if [ -n "$pids" ]; then
            kill $pids 2>/dev/null || true
        fi
    done
}


stop_caption_servers() {
    if [ "${CAPTION_SERVERS_OWNED:-0}" != "1" ]; then
        return 0
    fi
    _kill_owned_ports "$CAPTION_OWNED_PORTS"
    _wait_for_stopped_ports "127.0.0.1" "$CAPTION_OWNED_PORTS" || true
    CAPTION_SERVERS_OWNED=0
    CAPTION_OWNED_PORTS=""
}


stop_instruct_servers() {
    if [ "${INSTRUCT_SERVERS_OWNED:-0}" != "1" ]; then
        return 0
    fi
    _kill_owned_ports "$INSTRUCT_OWNED_PORTS"
    _wait_for_stopped_ports "127.0.0.1" "$INSTRUCT_OWNED_PORTS" || true
    INSTRUCT_SERVERS_OWNED=0
    INSTRUCT_OWNED_PORTS=""
}


cleanup_owned_services() {
    stop_caption_servers || true
    stop_instruct_servers || true
}
