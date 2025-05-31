#!/bin/bash

# =============================================================================
# Tac3D 触觉传感器启动脚本 (简化版)
# =============================================================================

# 脚本目录配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_DIR="$SCRIPT_DIR/main"
CONFIG_DIR="$SCRIPT_DIR/configs"
TAC3D_EXECUTABLE="$MAIN_DIR/Tac3D"

# 默认传感器
DEFAULT_SENSOR="AD2-0047L"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 日志函数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 显示帮助
show_help() {
    cat << EOF
Tac3D 触觉传感器启动脚本

用法: $0 [SENSOR_ID] [OPTIONS]

传感器ID:
    AD2-0047L    左手传感器 (默认)
    AD2-0046R    右手传感器

选项:
    -h, --help   显示帮助
    -l, --list   列出可用传感器
    -k, --kill   终止运行中的进程

示例:
    $0                # 启动默认传感器
    $0 AD2-0046R      # 启动右手传感器
    $0 --list         # 列出可用传感器
    $0 --kill         # 终止所有进程

EOF
}

# 列出可用传感器
list_sensors() {
    log_info "可用的传感器配置:"
    for sensor_dir in "$CONFIG_DIR"/*; do
        if [[ -d "$sensor_dir" ]]; then
            sensor_id=$(basename "$sensor_dir")
            echo "  - $sensor_id"
        fi
    done
}

# 终止进程
kill_processes() {
    log_info "终止Tac3D进程..."
    if pgrep -f "Tac3D" > /dev/null; then
        pkill -f "Tac3D"
        log_info "进程已终止"
    else
        log_info "没有运行中的进程"
    fi
}

# 基本检查
check_basic() {
    # 检查可执行文件
    if [[ ! -f "$TAC3D_EXECUTABLE" ]]; then
        log_error "找不到Tac3D: $TAC3D_EXECUTABLE"
        return 1
    fi
    
    # 检查执行权限
    if [[ ! -x "$TAC3D_EXECUTABLE" ]]; then
        log_error "Tac3D没有执行权限"
        return 1
    fi
    
    return 0
}

# 验证传感器配置
validate_sensor() {
    local sensor_id=$1
    local config_dir="$CONFIG_DIR/$sensor_id"
    
    if [[ ! -d "$config_dir" ]]; then
        log_error "传感器配置不存在: $sensor_id"
        log_info "可用配置:"
        list_sensors
        return 1
    fi
    
    if [[ ! -f "$config_dir/sensor.yaml" ]]; then
        log_error "配置文件缺失: $config_dir/sensor.yaml"
        return 1
    fi
    
    return 0
}

# 启动传感器
start_sensor() {
    local sensor_id=$1
    local config_dir="$CONFIG_DIR/$sensor_id"
    
    log_info "启动Tac3D传感器: $sensor_id"
    log_info "配置目录: $config_dir"
    
    # 切换到程序目录
    cd "$MAIN_DIR" || exit 1
    
    # 启动程序
    log_info "正在启动... (按Ctrl+C停止)"
    "$TAC3D_EXECUTABLE" -c "$config_dir"
}

# 主函数
main() {
    local sensor_id="$DEFAULT_SENSOR"
    
    # 解析参数
    case "${1:-}" in
        -h|--help)
            show_help
            exit 0
            ;;
        -l|--list)
            list_sensors
            exit 0
            ;;
        -k|--kill)
            kill_processes
            exit 0
            ;;
        "")
            # 使用默认传感器
            ;;
        *)
            sensor_id="$1"
            ;;
    esac
    
    # 基本检查
    if ! check_basic; then
        exit 1
    fi
    
    # 验证传感器配置
    if ! validate_sensor "$sensor_id"; then
        exit 1
    fi
    
    # 启动传感器
    start_sensor "$sensor_id"
}

# 运行主函数
main "$@"
