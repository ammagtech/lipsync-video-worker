#!/usr/bin/env bash
#
# install.sh
#
# Description: Installation script for the Augment VIP project
# This script sets up the necessary dependencies and configurations
#
# Usage: ./install.sh [options]
#   Options:
#     --help          Show this help message
#     --clean         Run database cleaning script after installation
#     --modify-ids    Run telemetry ID modification script after installation
#     --all           Run all scripts (clean and modify IDs)

set -e  # Exit immediately if a command exits with a non-zero status
set -u  # Treat unset variables as an error

# Text formatting
BOLD="\033[1m"
RED="\033[31m"
GREEN="\033[32m"
YELLOW="\033[33m"
BLUE="\033[34m"
RESET="\033[0m"

# Log functions
log_info() {
    echo -e "${BLUE}[INFO]${RESET} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${RESET} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${RESET} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${RESET} $1"
}

# Repository information
REPO_URL="https://raw.githubusercontent.com/azrilaiman2003/augment-vip/main"

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Determine if this is a standalone installation or from a cloned repository
if [[ "$SCRIPT_DIR" == *"/scripts" ]]; then
    # Script is in a scripts directory, likely a cloned repository
    PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
    STANDALONE_MODE=false
else
    # Script is not in a scripts directory, likely a standalone installation
    STANDALONE_MODE=true

    # Create a project directory for standalone installation
    PROJECT_ROOT="$( cd "$SCRIPT_DIR" && pwd )/augment-vip"
    log_info "Creating project directory at: $PROJECT_ROOT"
    mkdir -p "$PROJECT_ROOT"

    # Create scripts directory
    mkdir -p "$PROJECT_ROOT/scripts"
    SCRIPT_DIR="$PROJECT_ROOT/scripts"

    # Copy this script to the scripts directory
    cp "$0" "$SCRIPT_DIR/install.sh"
    chmod +x "$SCRIPT_DIR/install.sh"
fi

# Check for required system dependencies
check_dependencies() {
    log_info "Checking system dependencies..."

    local missing_deps=()

    # Check for common dependencies
    for cmd in sqlite3 curl jq; do
        if ! command -v "$cmd" &> /dev/null; then
            missing_deps+=("$cmd")
        fi
    done

    if [ ${#missing_deps[@]} -ne 0 ]; then
        log_warning "Missing dependencies: ${missing_deps[*]}"

        # Detect OS for installation instructions
        if [[ "$OSTYPE" == "darwin"* ]]; then
            log_info "To install on macOS, run: brew install ${missing_deps[*]}"
        elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
            log_info "To install on Ubuntu/Debian, run: sudo apt install ${missing_deps[*]}"
            log_info "To install on Fedora/RHEL, run: sudo dnf install ${missing_deps[*]}"
        elif [[ "$OSTYPE" == "msys"* || "$OSTYPE" == "cygwin"* ]]; then
            log_info "To install on Windows, we recommend using Chocolatey: choco install ${missing_deps[*]}"
        fi

        read -p "Do you want to continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_error "Installation aborted due to missing dependencies"
            exit 1
        fi
    else
        log_success "All system dependencies are installed"
    fi
}

# Make scripts executable
make_scripts_executable() {
    log_info "Making scripts executable..."

    find "$SCRIPT_DIR" -name "*.sh" -type f -exec chmod +x {} \;

    log_success "All scripts are now executable"
}

# Setup project configuration
setup_configuration() {
    log_info "Setting up project configuration..."

    # Create config directory if it doesn't exist
    mkdir -p "$PROJECT_ROOT/config"

    # Create default configuration file if it doesn't exist
    if [ ! -f "$PROJECT_ROOT/config/config.json" ]; then
        cat > "$PROJECT_ROOT/config/config.json" << EOF
{
    "version": "1.0.0",
    "environment": "development",
    "features": {
        "cleanCodeDb": true
    }
}
EOF
        log_success "Created default configuration file"
    else
        log_info "Configuration file already exists, skipping"
    fi
}

# Create necessary directories
create_directories() {
    log_info "Creating project directories..."

    # Create common directories
    mkdir -p "$PROJECT_ROOT/logs"
    mkdir -p "$PROJECT_ROOT/data"
    mkdir -p "$PROJECT_ROOT/temp"

    log_success "Project directories created"
}

# Download scripts from repository when in standalone mode
download_scripts() {
    if [ "$STANDALONE_MODE" = false ]; then
        log_info "Running in repository mode, skipping script download"
        return 0
    fi

    log_info "Running in standalone mode, downloading required scripts..."

    # List of scripts to download
    local scripts=("clean_code_db.sh" "id_modifier.sh")

    # Download each script
    for script in "${scripts[@]}"; do
        local script_url="$REPO_URL/scripts/$script"
        local script_path="$SCRIPT_DIR/$script"

        log_info "Downloading $script..."

        if curl -fsSL "$script_url" -o "$script_path"; then
            chmod +x "$script_path"
            log_success "Downloaded and made executable: $script"
        else
            log_error "Failed to download $script"
            return 1
        fi
    done

    log_success "All scripts downloaded successfully"
    return 0
}

# Run the database cleaning script
run_clean_script() {
    log_info "Running database cleaning script..."

    if [ -x "$SCRIPT_DIR/clean_code_db.sh" ]; then
        "$SCRIPT_DIR/clean_code_db.sh"
        log_success "Database cleaning completed"
    else
        log_error "Database cleaning script not found or not executable"
        return 1
    fi

    return 0
}

# Run the telemetry ID modification script
run_id_modifier_script() {
    log_info "Running telemetry ID modification script..."

    if [ -x "$SCRIPT_DIR/id_modifier.sh" ]; then
        "$SCRIPT_DIR/id_modifier.sh"
        log_success "Telemetry ID modification completed"
    else
        log_error "Telemetry ID modification script not found or not executable"
        return 1
    fi

    return 0
}

# Display help message
show_help() {
    echo "Augment VIP Installation Script"
    echo
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --help          Show this help message"
    echo "  --clean         Run database cleaning script after installation"
    echo "  --modify-ids    Run telemetry ID modification script after installation"
    echo "  --all           Run all scripts (clean and modify IDs)"
    echo
    echo "Example: $0 --all"
}

# Main installation function
main() {
    # Parse command line arguments
    local run_clean=false
    local run_modify_ids=false

    # If no arguments provided, just do the basic installation
    if [ $# -eq 0 ]; then
        log_info "Starting installation process for Augment VIP"
    else
        # Process arguments
        for arg in "$@"; do
            case $arg in
                --help)
                    show_help
                    exit 0
                    ;;
                --clean)
                    run_clean=true
                    ;;
                --modify-ids)
                    run_modify_ids=true
                    ;;
                --all)
                    run_clean=true
                    run_modify_ids=true
                    ;;
                *)
                    log_error "Unknown option: $arg"
                    show_help
                    exit 1
                    ;;
            esac
        done

        log_info "Starting installation process for Augment VIP with additional options"
    fi

    # Check dependencies
    check_dependencies

    # Download scripts if in standalone mode
    download_scripts

    # Make scripts executable
    make_scripts_executable

    # Setup configuration
    setup_configuration

    # Create directories
    create_directories

    log_success "Installation completed successfully!"

    # Check if any command line options were provided
    if [ "$run_clean" = true ] || [ "$run_modify_ids" = true ]; then
        # Run scripts based on command line options
        if [ "$run_clean" = true ]; then
            run_clean_script
        fi

        if [ "$run_modify_ids" = true ]; then
            run_id_modifier_script
        fi
    else
        # If no command line options were provided, ask the user if they want to run the scripts
        echo
        read -p "Would you like to clean VS Code databases now? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            run_clean_script
        fi

        echo
        read -p "Would you like to modify VS Code telemetry IDs now? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            run_id_modifier_script
        fi
    fi

    log_info "You can now use the scripts in the scripts directory"
    log_info "For example:"
    log_info "  - To clean VS Code databases: $SCRIPT_DIR/clean_code_db.sh"
    log_info "  - To modify telemetry IDs: $SCRIPT_DIR/id_modifier.sh"
}

# Execute main function
main "$@"
