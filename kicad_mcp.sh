#!/bin/bash
# Init-style controller for the built KiCad Copilot stdio MCP server.
# IMPORTANT: start stays in the foreground, preserving MCP stdin/stdout.
# Configure an MCP client to run: /bin/bash /absolute/path/kicad_mcp.sh start
# stop/status control only the instance started by this script, not KiCad GUI.
set -u
umask 077
NODE='/home/intl/.nvm/versions/node/v22.17.1/bin/node'
SERVER='/home/intl/wrk/dlab/kicad-copilot/dist/index.js'
KICAD='/home/intl/wrk/tools/kicad-10.0.1-1-x86_64.AppImage'
export KICAD_CLI_PATH='/home/intl/wrk/dlab/ai-bash-agent/kicad-cli'
PIDFILE='/home/intl/wrk/dlab/ai-bash-agent/kicad_mcp.pid'
LOCKFILE='/home/intl/wrk/dlab/ai-bash-agent/kicad_mcp.lock'

# Check command identity, not merely existence of a possibly reused PID.
running() {
    local -a argv=()
    pid=''
    [[ -r "$PIDFILE" ]] || return 1
    read -r pid < "$PIDFILE" || return 1
    [[ "$pid" =~ ^[1-9][0-9]*$ ]] || return 1
    [[ -r "/proc/$pid/cmdline" ]] || return 1
    mapfile -d '' -t argv < "/proc/$pid/cmdline" 2>/dev/null || return 1
    [[ "${argv[0]:-}" == "$NODE" && "${argv[1]:-}" == "$SERVER" ]] || return 1
    kill -0 "$pid" 2>/dev/null
}

case "${1:-}" in
    start)
        for executable in "$NODE" "$KICAD" "$KICAD_CLI_PATH"; do
            if [[ ! -x "$executable" ]]; then
                printf 'Not executable: %s\n' "$executable" >&2
                exit 1
            fi
        done
        [[ -r "$SERVER" ]] || { echo "Missing built server: $SERVER" >&2; exit 1; }
        command -v flock >/dev/null || { echo 'flock is required.' >&2; exit 1; }
        # Keep the lock across exec, preventing simultaneous starts.
        exec 9>"$LOCKFILE"
        flock -n 9 || { echo 'KiCad MCP already running (or starting).' >&2; exit 1; }
        if running; then
            echo "KiCad MCP already running (PID $pid)." >&2
            exit 1
        fi
        printf '%s\n' "$$" > "$PIDFILE"
        echo "Starting KiCad MCP (PID $$, foreground stdio)." >&2
        # No cd: preserve the MCP client's working directory for project paths.
        # All diagnostic output above goes to stderr; stdout is MCP only.
        exec "$NODE" "$SERVER"
        ;;
    stop)
        if ! running; then
            echo 'KiCad MCP is not running under this controller.'
            exit 0
        fi
        target="$pid"
        kill -TERM "$target" || exit 1
        for ((i=0; i<50; i++)); do
            if ! running || [[ "$pid" != "$target" ]]; then
                echo 'KiCad MCP stopped.'
                exit 0
            fi
            sleep 0.1
        done
        echo "PID $target did not stop within 5 seconds; no forced kill performed." >&2
        exit 1
        ;;
    status)
        if running; then
            echo "KiCad MCP running (PID $pid, stdio)."
            exit 0
        fi
        echo 'KiCad MCP stopped (or not managed by this controller).'
        exit 3
        ;;
    *)
        echo "Usage: /bin/bash $0 {start|stop|status}" >&2
        exit 2
        ;;
esac
