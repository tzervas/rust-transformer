#!/bin/bash

# Secure container execution script with monitoring
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS_DIR="${PROJECT_DIR}/logs"
CONTAINER_NAME="rust-transformer-demo"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Cleanup function
cleanup() {
    log "Cleaning up containers and resources..."
    
    # Stop and remove containers
    docker-compose down --remove-orphans --volumes 2>/dev/null || true
    
    # Remove any dangling containers
    docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
    docker rm -f "transformer-monitor" 2>/dev/null || true
    
    # Clean up any orphaned networks
    docker network prune -f 2>/dev/null || true
    
    success "Cleanup completed"
}

# Set trap for cleanup on script exit
trap cleanup EXIT INT TERM

# Pre-execution security checks
security_check() {
    log "Performing security checks..."
    
    # Check if Docker is running
    if ! docker info >/dev/null 2>&1; then
        error "Docker is not running or accessible"
        exit 1
    fi
    
    # Check available system resources
    MEMORY_MB=$(free -m | awk 'NR==2{print $7}')
    if [ "$MEMORY_MB" -lt 1024 ]; then
        warn "Low available memory: ${MEMORY_MB}MB"
    fi
    
    # Check disk space
    DISK_USAGE=$(df / | awk 'NR==2{print $5}' | sed 's/%//')
    if [ "$DISK_USAGE" -gt 90 ]; then
        warn "High disk usage: ${DISK_USAGE}%"
    fi
    
    success "Security checks passed"
}

# Monitor container execution
monitor_execution() {
    local timeout=30
    local counter=0
    
    log "Starting container monitoring (timeout: ${timeout}s)..."
    
    # Wait for container to start
    while [ $counter -lt $timeout ]; do
        if docker ps --filter "name=$CONTAINER_NAME" --format "{{.Names}}" | grep -q "$CONTAINER_NAME"; then
            success "Container $CONTAINER_NAME is running"
            break
        fi
        sleep 1
        ((counter++))
    done
    
    if [ $counter -eq $timeout ]; then
        error "Container failed to start within timeout"
        return 1
    fi
    
    # Monitor container health and resource usage
    log "Monitoring container resource usage..."
    
    # Get container stats in background
    (
        sleep 2  # Wait for container to initialize
        docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}" "$CONTAINER_NAME" 2>/dev/null || true
    ) &
    
    # Wait for container to complete
    if docker wait "$CONTAINER_NAME" >/dev/null 2>&1; then
        local exit_code
        exit_code=$(docker inspect "$CONTAINER_NAME" --format='{{.State.ExitCode}}')
        
        if [ "$exit_code" -eq 0 ]; then
            success "Container executed successfully (exit code: $exit_code)"
        else
            error "Container execution failed (exit code: $exit_code)"
        fi
        
        return "$exit_code"
    else
        error "Failed to monitor container execution"
        return 1
    fi
}

# Collect execution artifacts
collect_artifacts() {
    log "Collecting execution artifacts..."
    
    mkdir -p "$LOGS_DIR"
    
    # Container logs
    if docker logs "$CONTAINER_NAME" >"$LOGS_DIR/container-stdout.log" 2>"$LOGS_DIR/container-stderr.log"; then
        success "Container logs saved to $LOGS_DIR/"
    else
        warn "Failed to collect container logs"
    fi
    
    # Container inspection
    if docker inspect "$CONTAINER_NAME" >"$LOGS_DIR/container-inspect.json" 2>/dev/null; then
        success "Container inspection saved"
    fi
    
    # System information
    {
        echo "=== Execution Summary ==="
        echo "Timestamp: $(date)"
        echo "Host: $(hostname)"
        echo "Docker Version: $(docker version --format '{{.Server.Version}}' 2>/dev/null || echo 'Unknown')"
        echo "Container Name: $CONTAINER_NAME"
        echo ""
        echo "=== Resource Usage ==="
        docker stats --no-stream --format "{{.Container}}: CPU {{.CPUPerc}}, Memory {{.MemUsage}}" "$CONTAINER_NAME" 2>/dev/null || echo "Stats unavailable"
    } >"$LOGS_DIR/execution-summary.txt"
    
    success "Artifacts collected in $LOGS_DIR/"
}

# Main execution flow
main() {
    log "Starting secure Rust Transformer execution"
    log "Project directory: $PROJECT_DIR"
    
    cd "$PROJECT_DIR"
    
    # Perform security checks
    security_check
    
    # Clean up any existing containers
    cleanup
    
    # Build and start containers
    log "Building Docker image..."
    if docker-compose build --no-cache; then
        success "Docker image built successfully"
    else
        error "Failed to build Docker image"
        exit 1
    fi
    
    # Start execution with output capture
    log "Starting containerized execution..."
    
    # Create logs directory with proper permissions
    mkdir -p "$LOGS_DIR"
    
    # Run the container and capture logs
    docker-compose up --detach
    
    # Give container a moment to start
    sleep 2
    
    # Monitor execution
    if monitor_execution; then
        success "Execution completed successfully"
        EXIT_CODE=0
    else
        error "Execution failed or timed out"
        EXIT_CODE=1
    fi
    
    # Collect artifacts regardless of success/failure
    collect_artifacts
    
    # Show execution summary
    log "=== EXECUTION SUMMARY ==="
    if [ -f "$LOGS_DIR/container-stdout.log" ]; then
        echo -e "${BLUE}--- Container Output ---${NC}"
        cat "$LOGS_DIR/container-stdout.log"
    fi
    
    if [ -f "$LOGS_DIR/container-stderr.log" ] && [ -s "$LOGS_DIR/container-stderr.log" ]; then
        echo -e "${YELLOW}--- Container Errors ---${NC}"
        cat "$LOGS_DIR/container-stderr.log"
    fi
    
    # Final status
    if [ $EXIT_CODE -eq 0 ]; then
        success "✅ Rust Transformer executed successfully in secure container"
        success "📁 Logs and artifacts available in: $LOGS_DIR/"
    else
        error "❌ Execution failed - check logs for details"
        error "📁 Debug information available in: $LOGS_DIR/"
    fi
    
    return $EXIT_CODE
}

# Run main function
main "$@"