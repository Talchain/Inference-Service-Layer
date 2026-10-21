#!/bin/zsh
# Start/stop the pinned local PLoT->ISL pair. Stops by recorded PID only.
set -eu
ROOT=${T3_ROOT:-/private/tmp/track3-engine}
HERE=$(cd "$(dirname "$0")" && pwd)
ISL_PORT=${ISL_PORT:-18731}; PLOT_PORT=${PLOT_PORT:-18732}; PROXY_PORT=${PROXY_PORT:-18733}
CAPTURE=${CAPTURE:-$ROOT/logs/capture}
PLOT_DIR=${PLOT_DIR:-$ROOT/plot}
mkdir -p "$ROOT/logs"
case "${1:-}" in
start)
  (cd "$ROOT/isl" && ISL_AUTH_DISABLED=true ENVIRONMENT=development SENTRY_ENABLED=false REDIS_ENABLED=false \
     nohup "$ROOT/isl-venv/bin/python" -m uvicorn src.api.main:app --host 127.0.0.1 --port "$ISL_PORT" \
     > "$ROOT/logs/isl.log" 2>&1 & echo $! > "$ROOT/logs/isl.pid")
  nohup /usr/bin/python3 -I "$HERE/track3/capture_proxy.py" "http://127.0.0.1:$ISL_PORT" "$PROXY_PORT" "$CAPTURE" \
     > "$ROOT/logs/proxy.log" 2>&1 & echo $! > "$ROOT/logs/proxy.pid"
  for i in {1..60}; do curl -sf "http://127.0.0.1:$PROXY_PORT/health" >/dev/null 2>&1 && break; sleep 1; done
  (cd "$PLOT_DIR" && PORT="$PLOT_PORT" HOST=127.0.0.1 NODE_ENV=development ISL_ENABLE=1 \
     ISL_BASE_URL="http://127.0.0.1:$PROXY_PORT" AUTH_ENABLED=0 RATE_LIMIT_ENABLED=0 ISL_API_KEY=track3-local-unused TOKEN_HMAC_SECRET="$(openssl rand -hex 32)" \
     nohup ./node_modules/.bin/tsx src/main.ts > "$ROOT/logs/plot.log" 2>&1 & echo $! > "$ROOT/logs/plot.pid")
  for i in {1..90}; do
    curl -sf "http://127.0.0.1:$ISL_PORT/health" >/dev/null 2>&1 && curl -sf "http://127.0.0.1:$PLOT_PORT/health" >/dev/null 2>&1 && { echo "up after ${i}s"; exit 0; }
    sleep 1
  done
  echo "engine did not come up"; tail -20 "$ROOT/logs/isl.log" "$ROOT/logs/plot.log"; exit 1 ;;
stop)
  for f in isl plot proxy; do
    if [ -f "$ROOT/logs/$f.pid" ]; then pid=$(cat "$ROOT/logs/$f.pid"); pkill -TERM -P "$pid" 2>/dev/null || true; kill "$pid" 2>/dev/null || true; rm -f "$ROOT/logs/$f.pid"; fi
  done; echo stopped ;;
*) echo "usage: engine.sh start|stop"; exit 2 ;;
esac
