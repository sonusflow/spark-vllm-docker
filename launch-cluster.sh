#!/bin/bash

# Default Configuration
IMAGE_NAME="vllm-node"
DEFAULT_CONTAINER_NAME="vllm_node"
# Modify these if you want to pass additional docker args or set VLLM_SPARK_EXTRA_DOCKER_ARGS variable
DOCKER_ARGS="-e NCCL_IGNORE_CPU_AFFINITY=1 -v $HOME/.cache/huggingface:/root/.cache/huggingface"

# Append additional arguments from environment variable
if [[ -n "$VLLM_SPARK_EXTRA_DOCKER_ARGS" ]]; then
    DOCKER_ARGS="$DOCKER_ARGS $VLLM_SPARK_EXTRA_DOCKER_ARGS"
fi

# ETH_IF and IB_IF will be auto-detected if not provided
ETH_IF=""
IB_IF=""
NCCL_DEBUG_VAL=""

# Initialize variables
NODES_ARG=""
CONTAINER_NAME="$DEFAULT_CONTAINER_NAME"
COMMAND_TO_RUN=""
DAEMON_MODE="false"
CHECK_CONFIG="false"
ACTION="start"
CLUSTER_WAS_RUNNING="false"

# Function to print usage
usage() {
    echo "Usage: $0 [-n <node_ips>] [-t <image_name>] [--name <container_name>] [--eth-if <if_name>] [--ib-if <if_name>] [--nccl-debug <level>] [--check-config] [-d] [action] [command]"
    echo "  -n, --nodes     Comma-separated list of node IPs (Optional, auto-detected if omitted)"
    echo "  -t              Docker image name (Optional, default: $IMAGE_NAME)"
    echo "  --name          Container name (Optional, default: $DEFAULT_CONTAINER_NAME)"
    echo "  --eth-if        Ethernet interface (Optional, auto-detected)"
    echo "  --ib-if         InfiniBand interface (Optional, auto-detected)"
    echo "  --nccl-debug    NCCL debug level (Optional, one of: VERSION, WARN, INFO, TRACE). If no level is provided, defaults to INFO."
    echo "  --check-config  Check configuration and auto-detection without launching"
    echo "  -d              Daemon mode (only for 'start' action)"
    echo "  action          start | stop | status | exec (Default: start)"
    echo "  command         Command to run (only for 'exec' action)"
    exit 1
}

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -n|--nodes) NODES_ARG="$2"; shift ;;
        -t) IMAGE_NAME="$2"; shift ;;
        --name) CONTAINER_NAME="$2"; shift ;;
        --eth-if) ETH_IF="$2"; shift ;;
        --ib-if) IB_IF="$2"; shift ;;
        --nccl-debug)
            if [[ -n "$2" && "$2" =~ ^(VERSION|WARN|INFO|TRACE)$ ]]; then
                NCCL_DEBUG_VAL="$2"
                shift
            else
                NCCL_DEBUG_VAL="INFO"
            fi
            ;;
        --check-config) CHECK_CONFIG="true" ;;
        -d) DAEMON_MODE="true" ;;
        -h|--help) usage ;;
        start|stop|status) 
            ACTION="$1" 
            ;;
        exec)
            ACTION="exec"
            shift
            COMMAND_TO_RUN="$@"
            break
            ;;
        *) 
            # If it's not a flag and not a known action, treat as exec command for backward compatibility
            # unless it's the default 'start' implied.
            # However, to support "omitted" = start, we need to be careful.
            # If the arg looks like a command, it's exec.
            ACTION="exec"
            COMMAND_TO_RUN="$@"
            break 
            ;;
    esac
    shift
done

# Append NCCL_DEBUG if set, with validation
if [[ -n "$NCCL_DEBUG_VAL" ]]; then
    case "$NCCL_DEBUG_VAL" in
        VERSION|WARN|INFO|TRACE)
            DOCKER_ARGS="$DOCKER_ARGS -e NCCL_DEBUG=$NCCL_DEBUG_VAL"
            ;;
        *)
            echo "Error: Invalid value for --nccl-debug: $NCCL_DEBUG_VAL"
            echo "Allowed values: VERSION, WARN, INFO, TRACE"
            exit 1
            ;;
    esac
fi

# --- Auto-Detection Logic ---

# Source autodiscover module
source "$(dirname "$0")/autodiscover.sh"

# Perform auto-detection
detect_interfaces || exit 1
detect_nodes || exit 1

if [[ -z "$NODES_ARG" ]]; then
    echo "Error: Nodes argument (-n) is mandatory or could not be auto-detected."
    usage
fi

# Split nodes into array
IFS=',' read -r -a ALL_NODES <<< "$NODES_ARG"

# Detect Head IP (Local IP)
detect_local_ip || exit 1
HEAD_IP="$LOCAL_IP"

# Verify HEAD_IP is in ALL_NODES
FOUND_HEAD=false
for ip in "${ALL_NODES[@]}"; do
    ip=$(echo "$ip" | xargs)
    if [[ "$ip" == "$HEAD_IP" ]]; then
        FOUND_HEAD=true
        break
    fi
done

if [ "$FOUND_HEAD" = false ]; then
    echo "Error: Local IP ($HEAD_IP) is not in the list of nodes ($NODES_ARG)."
    exit 1
fi

echo "Head Node: $HEAD_IP"
echo "Worker Nodes: ${PEER_NODES[*]}"
echo "Container Name: $CONTAINER_NAME"
echo "Action: $ACTION"

# Check SSH connectivity to worker nodes
if [[ "$ACTION" == "start" || "$ACTION" == "exec" || "$CHECK_CONFIG" == "true" ]]; then
    if [ ${#WORKER_NODES[@]} -gt 0 ]; then
        echo "Checking SSH connectivity to worker nodes..."
        for worker in "${WORKER_NODES[@]}"; do
            if ! ssh -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=no "$worker" true 2>/dev/null; then
                echo "Error: Passwordless SSH to $worker failed."
                echo "  Please ensure SSH keys are configured and the host is reachable."
                exit 1
            fi
            echo "  SSH to $worker: OK"
        done
    fi
fi

if [[ "$CHECK_CONFIG" == "true" ]]; then
    echo "Configuration Check Complete."
    echo "  Image Name: $IMAGE_NAME"
    echo "  ETH Interface: $ETH_IF"
    echo "  IB Interface: $IB_IF"
    exit 0
fi

# Cleanup Function
cleanup() {
    # Remove traps to prevent nested cleanup
    trap - EXIT INT TERM HUP

    if [[ "$CLUSTER_WAS_RUNNING" == "true" ]]; then
        echo "Cluster was already running when script started. Skipping cleanup."
        return
    fi

    echo ""
    echo "Stopping cluster..."
    
    # Stop Head
    echo "Stopping head node ($HEAD_IP)..."
    docker stop "$CONTAINER_NAME" >/dev/null 2>&1 || true
    
    # Stop Workers
    for worker in "${WORKER_NODES[@]}"; do
        echo "Stopping worker node ($worker)..."
        ssh "$worker" "docker stop $CONTAINER_NAME" >/dev/null 2>&1 || true
    done
    
    echo "Cluster stopped."
}

# Handle 'stop' action
if [[ "$ACTION" == "stop" ]]; then
    cleanup
    exit 0
fi

# Handle 'status' action
if [[ "$ACTION" == "status" ]]; then
    echo "Checking status..."
    
    # Check Head
    if docker ps | grep -q "$CONTAINER_NAME"; then
        echo "[HEAD] $HEAD_IP: Container '$CONTAINER_NAME' is RUNNING."
        echo "--- Ray Status ---"
        docker exec "$CONTAINER_NAME" ray status || echo "Failed to get ray status."
        echo "------------------"
    else
        echo "[HEAD] $HEAD_IP: Container '$CONTAINER_NAME' is NOT running."
    fi
    
    # Check Workers
    for worker in "${WORKER_NODES[@]}"; do
        if ssh "$worker" "docker ps | grep -q '$CONTAINER_NAME'"; then
             echo "[WORKER] $worker: Container '$CONTAINER_NAME' is RUNNING."
        else
             echo "[WORKER] $worker: Container '$CONTAINER_NAME' is NOT running."
        fi
    done
    exit 0
fi

# Trap signals
# Only trap if we are NOT in daemon mode, OR if we are in exec mode (always cleanup after exec)
if [[ "$DAEMON_MODE" == "false" ]] || [[ "$ACTION" == "exec" ]]; then
    trap cleanup EXIT INT TERM HUP
fi

# Check if cluster is already running
check_cluster_running() {
    local running=false
    
    # Check Head
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "Warning: Container '$CONTAINER_NAME' is already running on head node ($HEAD_IP)."
        running=true
    fi
    
    # Check Workers
    for worker in "${WORKER_NODES[@]}"; do
        if ssh "$worker" "docker ps --format '{{.Names}}' | grep -q '^${CONTAINER_NAME}$'"; then
             echo "Warning: Container '$CONTAINER_NAME' is already running on worker node ($worker)."
             running=true
        fi
    done
    
    if [[ "$running" == "true" ]]; then
        echo "Cluster containers are already running. Skipping launch."
        CLUSTER_WAS_RUNNING="true"
        return 0
    fi
}

# Start Cluster Function
start_cluster() {
    check_cluster_running

    if [[ "$CLUSTER_WAS_RUNNING" == "true" ]]; then
        return
    fi

    # Start Head Node
    echo "Starting Head Node on $HEAD_IP..."
    docker run -d --privileged --gpus all --rm \
        --ipc=host --network host \
        --name "$CONTAINER_NAME" \
        $DOCKER_ARGS \
        "$IMAGE_NAME" \
        ./run-cluster-node.sh \
        --role head \
        --host-ip "$HEAD_IP" \
        --eth-if "$ETH_IF" \
        --ib-if "$IB_IF"

    # Start Worker Nodes
    # Start Worker Nodes
    for worker in "${WORKER_NODES[@]}"; do
        echo "Starting Worker Node on $worker..."
        ssh "$worker" "docker run -d --privileged --gpus all --rm \
            --ipc=host --network host \
            --name $CONTAINER_NAME \
            $DOCKER_ARGS \
            $IMAGE_NAME \
            ./run-cluster-node.sh \
            --role node \
            --host-ip $worker \
            --eth-if $ETH_IF \
            --ib-if $IB_IF \
            --head-ip $HEAD_IP"
    done

    wait_for_cluster
}

# Wait for Cluster Readiness
wait_for_cluster() {
    echo "Waiting for cluster to be ready..."
    local retries=30
    local count=0
    
    while [[ $count -lt $retries ]]; do
        # Check if ray is responsive
        if docker exec "$CONTAINER_NAME" ray status >/dev/null 2>&1; then
             echo "Cluster head is responsive."
             # Give workers a moment to connect
             sleep 5
             return 0
        fi
        
        sleep 2
        ((count++))
    done
    
    echo "Timeout waiting for cluster to start."
    exit 1
}

if [[ "$ACTION" == "exec" ]]; then
    start_cluster
    echo "Executing command on head node: $COMMAND_TO_RUN"
    
    # Check if running in a TTY to avoid "input device is not a TTY" error
    if [ -t 0 ]; then
        DOCKER_EXEC_FLAGS="-it"
    else
        DOCKER_EXEC_FLAGS="-i"
    fi
    
    docker exec $DOCKER_EXEC_FLAGS "$CONTAINER_NAME" bash -i -c "$COMMAND_TO_RUN"
elif [[ "$ACTION" == "start" ]]; then
    start_cluster
    if [[ "$DAEMON_MODE" == "true" ]]; then
        echo "Cluster started in background (Daemon mode)."
    else
        echo "Cluster started. Tailing logs from head node..."
        echo "Press Ctrl+C to stop the cluster."
        docker logs -f "$CONTAINER_NAME" &
        wait $!
    fi
fi
