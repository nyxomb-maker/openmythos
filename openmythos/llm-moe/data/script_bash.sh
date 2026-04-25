# Bash / Shell Scripting Training Data
# System administration, automation, and DevOps patterns.

#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# ═══════════════════════════════════════════════════════════════════════
# Constants & Configuration
# ═══════════════════════════════════════════════════════════════════════

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_NAME="$(basename "$0")"
readonly LOG_FILE="/var/log/${SCRIPT_NAME%.sh}.log"
readonly LOCK_FILE="/tmp/${SCRIPT_NAME%.sh}.lock"

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[0;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

# ═══════════════════════════════════════════════════════════════════════
# Logging Functions
# ═══════════════════════════════════════════════════════════════════════

log_info()  { echo -e "${BLUE}[INFO]${NC}  $(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG_FILE"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG_FILE" >&2; }
log_error() { echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG_FILE" >&2; }
log_ok()    { echo -e "${GREEN}[OK]${NC}    $(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG_FILE"; }

# ═══════════════════════════════════════════════════════════════════════
# Error Handling & Cleanup
# ═══════════════════════════════════════════════════════════════════════

cleanup() {
    local exit_code=$?
    rm -f "$LOCK_FILE"
    if [[ $exit_code -ne 0 ]]; then
        log_error "Script exited with code $exit_code"
    fi
    exit "$exit_code"
}

trap cleanup EXIT ERR INT TERM

# Prevent concurrent execution
acquire_lock() {
    if [[ -f "$LOCK_FILE" ]]; then
        local pid
        pid=$(cat "$LOCK_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            log_error "Another instance is running (PID: $pid)"
            exit 1
        else
            log_warn "Stale lock file found, removing"
            rm -f "$LOCK_FILE"
        fi
    fi
    echo $$ > "$LOCK_FILE"
}

# ═══════════════════════════════════════════════════════════════════════
# Argument Parsing
# ═══════════════════════════════════════════════════════════════════════

usage() {
    cat <<EOF
Usage: $SCRIPT_NAME [OPTIONS] <command>

Commands:
    deploy          Deploy application to target environment
    rollback        Rollback to previous version
    status          Check deployment status
    backup          Create database backup
    restore         Restore from backup

Options:
    -e, --env       Environment (dev|staging|prod) [default: dev]
    -v, --version   Application version to deploy
    -d, --dry-run   Show what would be done without executing
    -f, --force     Skip confirmation prompts
    -h, --help      Show this help message
    --verbose       Enable verbose output

Examples:
    $SCRIPT_NAME deploy -e prod -v 2.3.1
    $SCRIPT_NAME rollback -e staging
    $SCRIPT_NAME backup --env prod

EOF
}

# Parse arguments
ENV="dev"
VERSION=""
DRY_RUN=false
FORCE=false
VERBOSE=false
COMMAND=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -e|--env)
            ENV="$2"
            shift 2
            ;;
        -v|--version)
            VERSION="$2"
            shift 2
            ;;
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -f|--force)
            FORCE=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        deploy|rollback|status|backup|restore)
            COMMAND="$1"
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# ═══════════════════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════════════════

confirm() {
    if [[ "$FORCE" == true ]]; then
        return 0
    fi
    local prompt="${1:-Are you sure?}"
    read -rp "$prompt [y/N] " response
    [[ "$response" =~ ^[yY]$ ]]
}

run_cmd() {
    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY RUN] $*"
        return 0
    fi
    if [[ "$VERBOSE" == true ]]; then
        log_info "Running: $*"
    fi
    eval "$@"
}

# Check required tools
check_dependencies() {
    local deps=("docker" "kubectl" "jq" "curl" "openssl")
    local missing=()

    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &>/dev/null; then
            missing+=("$dep")
        fi
    done

    if [[ ${#missing[@]} -gt 0 ]]; then
        log_error "Missing dependencies: ${missing[*]}"
        exit 1
    fi
    log_ok "All dependencies satisfied"
}

# Wait for service health
wait_for_healthy() {
    local url="$1"
    local max_attempts="${2:-30}"
    local delay="${3:-5}"

    log_info "Waiting for $url to be healthy..."
    for ((i=1; i<=max_attempts; i++)); do
        if curl -sf "$url" &>/dev/null; then
            log_ok "Service is healthy after $i attempts"
            return 0
        fi
        echo -n "."
        sleep "$delay"
    done

    log_error "Service not healthy after $max_attempts attempts"
    return 1
}

# ═══════════════════════════════════════════════════════════════════════
# Deploy Command
# ═══════════════════════════════════════════════════════════════════════

cmd_deploy() {
    log_info "Deploying version ${VERSION} to ${ENV}"

    if [[ "$ENV" == "prod" ]]; then
        confirm "⚠️  You're deploying to PRODUCTION. Continue?" || exit 0
    fi

    # Build
    log_info "Building Docker image..."
    run_cmd docker build \
        --build-arg "VERSION=${VERSION}" \
        --build-arg "ENV=${ENV}" \
        -t "myapp:${VERSION}" \
        -t "myapp:latest" \
        "${SCRIPT_DIR}/.."

    # Push
    log_info "Pushing to registry..."
    run_cmd docker push "registry.example.com/myapp:${VERSION}"

    # Deploy to Kubernetes
    log_info "Updating Kubernetes deployment..."
    run_cmd kubectl set image \
        "deployment/myapp" \
        "myapp=registry.example.com/myapp:${VERSION}" \
        --namespace="${ENV}" \
        --record

    # Wait for rollout
    log_info "Waiting for rollout to complete..."
    run_cmd kubectl rollout status \
        "deployment/myapp" \
        --namespace="${ENV}" \
        --timeout=300s

    # Health check
    local health_url
    case "$ENV" in
        dev)     health_url="http://dev.example.com/health" ;;
        staging) health_url="https://staging.example.com/health" ;;
        prod)    health_url="https://api.example.com/health" ;;
    esac
    wait_for_healthy "$health_url"

    log_ok "Deployment complete! Version ${VERSION} is live on ${ENV}"
}

# ═══════════════════════════════════════════════════════════════════════
# Backup Command
# ═══════════════════════════════════════════════════════════════════════

cmd_backup() {
    local timestamp
    timestamp=$(date '+%Y%m%d_%H%M%S')
    local backup_file="backup_${ENV}_${timestamp}.sql.gz"

    log_info "Creating backup for ${ENV} environment..."

    # Get database credentials from Kubernetes secret
    local db_host db_name db_user db_pass
    db_host=$(kubectl get secret db-credentials -n "${ENV}" -o jsonpath='{.data.host}' | base64 -d)
    db_name=$(kubectl get secret db-credentials -n "${ENV}" -o jsonpath='{.data.name}' | base64 -d)
    db_user=$(kubectl get secret db-credentials -n "${ENV}" -o jsonpath='{.data.user}' | base64 -d)
    db_pass=$(kubectl get secret db-credentials -n "${ENV}" -o jsonpath='{.data.password}' | base64 -d)

    # Dump and compress
    run_cmd "PGPASSWORD='${db_pass}' pg_dump \
        -h '${db_host}' \
        -U '${db_user}' \
        -d '${db_name}' \
        --format=custom \
        --compress=9 \
        --verbose \
        | gzip > '/backups/${backup_file}'"

    # Upload to S3
    run_cmd aws s3 cp \
        "/backups/${backup_file}" \
        "s3://myapp-backups/${ENV}/${backup_file}" \
        --storage-class STANDARD_IA

    # Cleanup old local backups (keep 7 days)
    find /backups -name "backup_${ENV}_*" -mtime +7 -delete

    local size
    size=$(du -sh "/backups/${backup_file}" 2>/dev/null | cut -f1)
    log_ok "Backup complete: ${backup_file} (${size})"
}

# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

main() {
    acquire_lock
    check_dependencies

    log_info "Command: ${COMMAND}, Environment: ${ENV}, Dry Run: ${DRY_RUN}"

    case "${COMMAND}" in
        deploy)   cmd_deploy ;;
        backup)   cmd_backup ;;
        status)   kubectl get pods -n "${ENV}" -l app=myapp ;;
        rollback) run_cmd kubectl rollout undo "deployment/myapp" -n "${ENV}" ;;
        *)
            log_error "Unknown command: ${COMMAND}"
            usage
            exit 1
            ;;
    esac
}

main "$@"
