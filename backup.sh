#!/bin/bash
# backup.sh - Backup Stock Rotation (FORTRESS SIDEWAYS)
#
# Usage:
#   ./backup.sh          # Lite backup (code/config/docs only)
#   ./backup.sh lite     # Lite backup (code/config/docs only)
#   ./backup.sh full     # Full backup (everything)
#
# Backups are stored in ~/backups/stock-rotation/

set -e

# Configuration
PROJECT_NAME="stock-rotation"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKUP_BASE_DIR="$HOME/backups/$PROJECT_NAME"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
echo_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
echo_error() { echo -e "${RED}[ERROR]${NC} $1"; }
echo_header() { echo -e "\n${BLUE}=== $1 ===${NC}"; }

# Parse mode argument
MODE="${1:-lite}"
if [[ "$MODE" != "full" && "$MODE" != "lite" ]]; then
    echo_error "Invalid mode: $MODE"
    echo "Usage: $0 [lite|full]"
    echo "  lite - Code, config, docs only (default)"
    echo "  full - Everything including data and logs"
    exit 1
fi

# Create backup directory
mkdir -p "$BACKUP_BASE_DIR"

# Set backup filename based on mode
if [[ "$MODE" == "full" ]]; then
    BACKUP_FILE="$BACKUP_BASE_DIR/${PROJECT_NAME}_full_${TIMESTAMP}.zip"
else
    BACKUP_FILE="$BACKUP_BASE_DIR/${PROJECT_NAME}_lite_${TIMESTAMP}.zip"
fi

echo_header "Stock Rotation (FORTRESS) - Backup Script"
echo_info "Project: $PROJECT_DIR"
echo_info "Mode: $MODE"
echo_info "Output: $BACKUP_FILE"

# Change to parent directory for clean paths in zip
cd "$(dirname "$PROJECT_DIR")"
PROJECT_BASENAME=$(basename "$PROJECT_DIR")

if [[ "$MODE" == "lite" ]]; then
    echo_header "Creating Lite Backup (code/config/docs)"

    # Lite backup: exclude large/generated directories
    # Include: Python code, config, docs, scripts, tests
    # Exclude: venv, __pycache__, .git, data, logs, .pytest_cache, *.pyc, etc.

    zip -r "$BACKUP_FILE" "$PROJECT_BASENAME" \
        -x "$PROJECT_BASENAME/venv/*" \
        -x "$PROJECT_BASENAME/.venv/*" \
        -x "$PROJECT_BASENAME/__pycache__/*" \
        -x "$PROJECT_BASENAME/*/__pycache__/*" \
        -x "$PROJECT_BASENAME/*/*/__pycache__/*" \
        -x "$PROJECT_BASENAME/*/*/*/__pycache__/*" \
        -x "$PROJECT_BASENAME/.git/*" \
        -x "$PROJECT_BASENAME/.git" \
        -x "$PROJECT_BASENAME/data/*" \
        -x "$PROJECT_BASENAME/logs/*" \
        -x "$PROJECT_BASENAME/.pytest_cache/*" \
        -x "$PROJECT_BASENAME/*/.pytest_cache/*" \
        -x "$PROJECT_BASENAME/.mypy_cache/*" \
        -x "$PROJECT_BASENAME/.ruff_cache/*" \
        -x "$PROJECT_BASENAME/*.egg-info/*" \
        -x "$PROJECT_BASENAME/*/*.egg-info/*" \
        -x "$PROJECT_BASENAME/.coverage" \
        -x "$PROJECT_BASENAME/htmlcov/*" \
        -x "$PROJECT_BASENAME/*.zip" \
        -x "$PROJECT_BASENAME/*.tar.gz" \
        -x "$PROJECT_BASENAME/*.bak" \
        -x "$PROJECT_BASENAME/*.pyc" \
        -x "$PROJECT_BASENAME/*/*.pyc" \
        -x "$PROJECT_BASENAME/*/*/*.pyc" \
        -x "$PROJECT_BASENAME/.DS_Store" \
        -x "$PROJECT_BASENAME/*/.DS_Store" \
        -x "$PROJECT_BASENAME/.env" \
        -x "$PROJECT_BASENAME/.kite_token*" \
        -x "$PROJECT_BASENAME/node_modules/*" \
        -x "$PROJECT_BASENAME/.claude/*" \
        -x "$PROJECT_BASENAME/*.parquet" \
        -x "$PROJECT_BASENAME/*/*.parquet" \
        | grep -v "adding:" || true

    echo_info "Lite backup excludes: venv, .git, data, logs, cache, .env, tokens, parquet"

else
    echo_header "Creating Full Backup (everything)"

    # Full backup: include everything except truly temporary/sensitive files
    # Still exclude: venv (can be recreated), .git (use git for versioning), .env (secrets)

    zip -r "$BACKUP_FILE" "$PROJECT_BASENAME" \
        -x "$PROJECT_BASENAME/venv/*" \
        -x "$PROJECT_BASENAME/.venv/*" \
        -x "$PROJECT_BASENAME/__pycache__/*" \
        -x "$PROJECT_BASENAME/*/__pycache__/*" \
        -x "$PROJECT_BASENAME/*/*/__pycache__/*" \
        -x "$PROJECT_BASENAME/*/*/*/__pycache__/*" \
        -x "$PROJECT_BASENAME/.git/*" \
        -x "$PROJECT_BASENAME/.git" \
        -x "$PROJECT_BASENAME/.pytest_cache/*" \
        -x "$PROJECT_BASENAME/*/.pytest_cache/*" \
        -x "$PROJECT_BASENAME/.mypy_cache/*" \
        -x "$PROJECT_BASENAME/.ruff_cache/*" \
        -x "$PROJECT_BASENAME/*.egg-info/*" \
        -x "$PROJECT_BASENAME/*/*.egg-info/*" \
        -x "$PROJECT_BASENAME/*.zip" \
        -x "$PROJECT_BASENAME/*.tar.gz" \
        -x "$PROJECT_BASENAME/*.pyc" \
        -x "$PROJECT_BASENAME/*/*.pyc" \
        -x "$PROJECT_BASENAME/*/*/*.pyc" \
        -x "$PROJECT_BASENAME/.DS_Store" \
        -x "$PROJECT_BASENAME/*/.DS_Store" \
        -x "$PROJECT_BASENAME/.env" \
        -x "$PROJECT_BASENAME/.kite_token*" \
        -x "$PROJECT_BASENAME/node_modules/*" \
        -x "$PROJECT_BASENAME/.claude/*" \
        | grep -v "adding:" || true

    echo_info "Full backup includes: data, logs (excludes: venv, .git, .env, tokens)"
fi

# Return to project directory
cd "$PROJECT_DIR"

# Show backup info
echo_header "Backup Complete"

if [[ -f "$BACKUP_FILE" ]]; then
    BACKUP_SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
    FILE_COUNT=$(unzip -l "$BACKUP_FILE" 2>/dev/null | tail -1 | awk '{print $2}')

    echo_info "File: $BACKUP_FILE"
    echo_info "Size: $BACKUP_SIZE"
    echo_info "Files: $FILE_COUNT"

    # List recent backups
    echo_header "Recent Backups"
    ls -lht "$BACKUP_BASE_DIR"/*.zip 2>/dev/null | head -5 | while read line; do
        echo "  $line"
    done

    # Show disk usage of backup directory
    TOTAL_SIZE=$(du -sh "$BACKUP_BASE_DIR" 2>/dev/null | cut -f1)
    BACKUP_COUNT=$(ls -1 "$BACKUP_BASE_DIR"/*.zip 2>/dev/null | wc -l)
    echo ""
    echo_info "Backup directory: $BACKUP_BASE_DIR"
    echo_info "Total backups: $BACKUP_COUNT files, $TOTAL_SIZE"

    # Cleanup hint
    OLD_COUNT=$(find "$BACKUP_BASE_DIR" -name "*.zip" -mtime +30 2>/dev/null | wc -l)
    if [[ $OLD_COUNT -gt 0 ]]; then
        echo_warn "Found $OLD_COUNT backups older than 30 days"
        echo_info "To clean up old backups: find $BACKUP_BASE_DIR -name '*.zip' -mtime +30 -delete"
    fi
else
    echo_error "Backup file not created!"
    exit 1
fi

echo ""
echo_info "Restore with: unzip $BACKUP_FILE -d /path/to/restore/"
