#!/bin/bash
#
# FORTRESS SIDEWAYS - Startup Script
#
# This script ensures the virtual environment and dependencies are set up,
# then launches the interactive CLI.
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"
REQUIREMENTS_HASH_FILE="$VENV_DIR/.requirements_hash"

echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════╗"
echo "║          FORTRESS SIDEWAYS - Startup                     ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to get hash of pyproject.toml for dependency tracking
get_requirements_hash() {
    if [ -f "$SCRIPT_DIR/pyproject.toml" ]; then
        md5sum "$SCRIPT_DIR/pyproject.toml" 2>/dev/null | cut -d' ' -f1 || echo "none"
    else
        echo "none"
    fi
}

# Check Python version
check_python() {
    echo -e "${YELLOW}Checking Python...${NC}"

    # Try python3 first, then python
    if command_exists python3; then
        PYTHON_CMD="python3"
    elif command_exists python; then
        PYTHON_CMD="python"
    else
        echo -e "${RED}Error: Python not found. Please install Python 3.10+${NC}"
        exit 1
    fi

    # Check version
    PYTHON_VERSION=$($PYTHON_CMD -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    PYTHON_MAJOR=$($PYTHON_CMD -c 'import sys; print(sys.version_info.major)')
    PYTHON_MINOR=$($PYTHON_CMD -c 'import sys; print(sys.version_info.minor)')

    if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
        echo -e "${RED}Error: Python 3.10+ required. Found: $PYTHON_VERSION${NC}"
        exit 1
    fi

    echo -e "${GREEN}✓ Python $PYTHON_VERSION found${NC}"
}

# Create virtual environment if it doesn't exist
setup_venv() {
    if [ ! -d "$VENV_DIR" ]; then
        echo -e "${YELLOW}Creating virtual environment...${NC}"
        $PYTHON_CMD -m venv "$VENV_DIR"
        echo -e "${GREEN}✓ Virtual environment created${NC}"
    else
        echo -e "${GREEN}✓ Virtual environment exists${NC}"
    fi
}

# Activate virtual environment
activate_venv() {
    echo -e "${YELLOW}Activating virtual environment...${NC}"

    if [ -f "$VENV_DIR/bin/activate" ]; then
        source "$VENV_DIR/bin/activate"
        echo -e "${GREEN}✓ Virtual environment activated${NC}"
    else
        echo -e "${RED}Error: Could not find venv activation script${NC}"
        exit 1
    fi
}

# Install/update dependencies
install_deps() {
    CURRENT_HASH=$(get_requirements_hash)
    STORED_HASH=""

    if [ -f "$REQUIREMENTS_HASH_FILE" ]; then
        STORED_HASH=$(cat "$REQUIREMENTS_HASH_FILE")
    fi

    # Check if fortress is installed
    if ! python -c "import fortress" 2>/dev/null; then
        NEEDS_INSTALL=true
    elif [ "$CURRENT_HASH" != "$STORED_HASH" ]; then
        NEEDS_INSTALL=true
        echo -e "${YELLOW}Dependencies changed, updating...${NC}"
    else
        NEEDS_INSTALL=false
    fi

    if [ "$NEEDS_INSTALL" = true ]; then
        echo -e "${YELLOW}Installing dependencies...${NC}"

        # Upgrade pip first
        pip install --upgrade pip --quiet

        # Install the package with dev dependencies
        pip install -e "$SCRIPT_DIR[dev]" --quiet

        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ Dependencies installed${NC}"
            echo "$CURRENT_HASH" > "$REQUIREMENTS_HASH_FILE"
        else
            echo -e "${RED}Error: Failed to install dependencies${NC}"
            exit 1
        fi
    else
        echo -e "${GREEN}✓ Dependencies up to date${NC}"
    fi
}

# Check for config file
check_config() {
    if [ ! -f "$SCRIPT_DIR/config.yaml" ]; then
        echo -e "${YELLOW}Warning: config.yaml not found${NC}"
        echo -e "${YELLOW}Creating default config file...${NC}"

        cat > "$SCRIPT_DIR/config.yaml" << 'EOF'
# FORTRESS SIDEWAYS Configuration
# Set your Zerodha API credentials below

zerodha:
  api_key: ""
  api_secret: ""

portfolio:
  initial_capital: 1600000
  max_positions: 20

rotation:
  top_sectors: 3
  stocks_per_sector: 5
  min_sector_stocks: 3
  min_rrv_threshold: 0.5
  rebalance_day: "friday"

momentum:
  lookback_return: 126
  lookback_volatility: 63
  trading_days_year: 252

risk:
  max_single_position: 0.08
  hard_max_position: 0.12
  max_sector_exposure: 0.35
  hard_max_sector: 0.45
  max_drawdown_warning: 0.15
  max_drawdown_halt: 0.25
  daily_loss_limit: 0.03
  vix_caution: 20.0
  vix_defensive: 25.0

costs:
  transaction_cost: 0.003

paths:
  universe_file: "stock-universe.json"
  log_dir: "logs"
  data_cache: ".cache"
EOF
        echo -e "${GREEN}✓ Default config created${NC}"
        echo -e "${YELLOW}Please edit config.yaml and add your Zerodha API credentials${NC}"
    else
        echo -e "${GREEN}✓ Config file exists${NC}"
    fi
}

# Check for universe file
check_universe() {
    if [ ! -f "$SCRIPT_DIR/stock-universe.json" ]; then
        echo -e "${RED}Error: stock-universe.json not found${NC}"
        echo -e "${RED}Please ensure the universe file is in: $SCRIPT_DIR${NC}"
        exit 1
    else
        echo -e "${GREEN}✓ Universe file exists${NC}"
    fi
}

# Main execution
main() {
    cd "$SCRIPT_DIR"

    # Load environment variables from .env if present
    if [ -f "$SCRIPT_DIR/.env" ]; then
        set -a
        source "$SCRIPT_DIR/.env"
        set +a
        echo -e "${GREEN}✓ Environment loaded from .env${NC}"
    fi

    check_python
    setup_venv
    activate_venv
    install_deps
    check_config
    check_universe

    echo ""
    echo -e "${GREEN}Starting FORTRESS SIDEWAYS...${NC}"
    echo ""

    # Run the CLI
    exec python -m fortress
}

# Handle Ctrl+C gracefully
trap 'echo -e "\n${YELLOW}Interrupted${NC}"; exit 0' INT

# Run main function
main "$@"
