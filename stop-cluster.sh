#!/bin/bash
# Stop vLLM cluster containers on all nodes
# Usage: ./stop-cluster.sh [node1 node2 ...]
# If no nodes specified, reads from .env file (CLUSTER_NODES)

CONTAINER_NAME="vllm_node"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Get nodes from CLI args, or .env, or fail
if [[ $# -gt 0 ]]; then
    NODES="$@"
elif [[ -f "$SCRIPT_DIR/.env" ]]; then
    source "$SCRIPT_DIR/.env"
    NODES="${CLUSTER_NODES//,/ }"
else
    echo "Usage: $0 [node1 node2 ...]"
    echo "Or create .env with CLUSTER_NODES=ip1,ip2,..."
    exit 1
fi

echo "Stopping vLLM cluster..."
for ip in $NODES; do
    echo -n "  $ip: "
    if ssh "$ip" "docker stop $CONTAINER_NAME && docker container remove $CONTAINER_NAME" 2>/dev/null; then
        echo "stopped"
    else
        echo "not running"
    fi
done
echo "Done."
