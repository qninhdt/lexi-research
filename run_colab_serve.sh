#!/usr/bin/env bash
# Host Qwen3.5-4B on a Colab GPU with vLLM and publish an OpenAI-compatible
# endpoint over a Cloudflare quick tunnel.
#
# Usage:
#   bash run_colab_serve.sh
#   MODEL_ID=Qwen/Qwen3.5-4B MAX_MODEL_LEN=16384 bash run_colab_serve.sh
#   API_KEY=my-secret bash run_colab_serve.sh
#   STOP=1 bash run_colab_serve.sh          # tear the session down
#
# Prints a public HTTPS base URL plus the bearer token to use with it. The
# server keeps running after this script exits; the session bills compute units
# until `colab stop -s "$SESSION"`.
#
# Requires: google-colab-cli (`uv tool install google-colab-cli`)

set -euo pipefail

GPU="${GPU:-L4}"
SESSION="${SESSION:-lexi-vllm}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-4B}"
SERVED_NAME="${SERVED_NAME:-lexi}"
PORT="${PORT:-8000}"
# 32768 is what fits alongside the weights on a 22 GiB L4: the measured run left
# 9.3 GiB for the KV cache, which is 273k tokens of paged cache — roughly 8x
# concurrency at full context. Raising this trades concurrency for reach.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
# vLLM version is pinned because Qwen3.5's `Qwen3_5ForConditionalGeneration` is
# only in the registry from 0.26.0 onward; older wheels reject the checkpoint.
VLLM_VERSION="${VLLM_VERSION:-0.26.0}"
# An empty key disables authentication. The tunnel URL is public and
# unguessable-but-not-secret, so the default sets one.
API_KEY="${API_KEY:-lexi-$(head -c 12 /dev/urandom | od -An -tx1 | tr -d ' \n')}"
READY_TIMEOUT="${READY_TIMEOUT:-900}"

for value_name in MODEL_ID SERVED_NAME SESSION; do
  value="${!value_name}"
  case "${value}" in
    *[!A-Za-z0-9_./-]*|"")
      echo "${value_name} must contain only letters, digits, '_', '.', '/', and '-'." >&2
      exit 2
      ;;
  esac
done
case "${API_KEY}" in
  *[!A-Za-z0-9_.-]*) echo "API_KEY must contain only letters, digits, '_', '.', and '-'." >&2; exit 2 ;;
esac
for value_name in PORT MAX_MODEL_LEN MAX_NUM_SEQS READY_TIMEOUT; do
  case "${!value_name}" in *[!0-9]*|""|0) echo "${value_name} must be a positive integer." >&2; exit 2 ;; esac
done

if [ "${STOP:-0}" = "1" ]; then
  colab stop -s "${SESSION}" && echo "Stopped session '${SESSION}'."
  exit 0
fi

echo "═══════════════════════════════════════════════════════"
echo "🚀  Lexi-Research — vLLM on Colab"
echo "    GPU: ${GPU}   Model: ${MODEL_ID}   Session: ${SESSION}"
echo "    Context: ${MAX_MODEL_LEN}   Concurrency: ${MAX_NUM_SEQS}   vLLM: ${VLLM_VERSION}"
echo "═══════════════════════════════════════════════════════"

# The VM is deliberately left running on every exit path: the whole point is a
# server that outlives this script. Failures print how to inspect and stop it
# rather than tearing down a VM that may have a working server on it.
cleanup() {
  status=$?
  [ "${status}" -eq 0 ] && return
  echo "" >&2
  echo "⚠️  Exited with status ${status}. Session '${SESSION}' is still RUNNING." >&2
  echo "    Logs:  colab exec -s ${SESSION} <<< \"import subprocess; print(subprocess.run(['tail','-40','/content/logs/vllm.log'],capture_output=True,text=True).stdout)\"" >&2
  echo "    Stop:  colab stop -s ${SESSION}" >&2
  echo "" >&2
  echo "    If 'colab sessions' shows the VM as an orphan ([?] with no name), its" >&2
  echo "    per-VM proxy token expired and the CLI pruned the local record while" >&2
  echo "    the VM kept running and billing. 'colab stop' cannot see it; release" >&2
  echo "    it server-side:" >&2
  echo "      bash ops/release-colab-orphans.sh" >&2
}
trap cleanup EXIT

# ── 1. Provision ──────────────────────────────────────────
echo ""
echo "--- [1/5] Provisioning Colab VM (GPU=${GPU}) ---"
if colab status -s "${SESSION}" >/dev/null 2>&1; then
  echo "Reusing live session '${SESSION}'."
else
  colab new -s "${SESSION}" --gpu "${GPU}"
fi

# ── 2. Install vLLM ───────────────────────────────────────
echo ""
echo "--- [2/5] Installing vLLM ${VLLM_VERSION} (several minutes on a cold VM) ---"
# Started detached, then polled from here in short calls. A cold install takes
# minutes, and holding one exec open that long loses the websocket
# ("Connection was lost") even though the install itself keeps running on the
# VM. Short calls reconnect each time, so a dropped socket costs one poll.
cat <<PYINSTALL | colab exec -s "${SESSION}" --timeout 120
import subprocess

have = subprocess.run("pip show vllm 2>/dev/null | grep -q 'Version: ${VLLM_VERSION}'", shell=True).returncode == 0
if have:
    print(">>> already installed", flush=True)
else:
    subprocess.run("mkdir -p /content/logs", shell=True)
    subprocess.run("setsid nohup pip install -q vllm==${VLLM_VERSION} "
                   "> /content/logs/install.log 2>&1 < /dev/null &", shell=True)
    print(">>> install started", flush=True)
PYINSTALL

install_deadline=$(( $(date +%s) + 1800 ))
while [ "$(date +%s)" -lt "${install_deadline}" ]; do
  install_state=$(cat <<PYPOLL | colab exec -s "${SESSION}" --timeout 120 2>/dev/null | tail -1
import subprocess, time
for _ in range(10):
    busy = subprocess.run("pgrep -f 'pip install' >/dev/null", shell=True).returncode == 0
    if not busy:
        break
    time.sleep(9)
done = subprocess.run("pip show vllm >/dev/null 2>&1", shell=True).returncode == 0
print("installed" if done else ("installing" if busy else "failed"), flush=True)
PYPOLL
)
  case "${install_state}" in
    installed) break ;;
    failed)
      echo "❌ vLLM install failed. Log:" >&2
      echo "import subprocess; print(subprocess.run(['tail','-20','/content/logs/install.log'],capture_output=True,text=True).stdout)" \
        | colab exec -s "${SESSION}" --timeout 60 >&2 || true
      exit 1
      ;;
    *) echo "    still installing" ;;
  esac
done
if [ "${install_state}" != "installed" ]; then
  echo "❌ vLLM install did not finish within 1800s." >&2
  exit 1
fi
echo "✅ vLLM ${VLLM_VERSION} present."

# ── 3. Launch the server ─────────────────────────────────
echo ""
echo "--- [3/5] Launching vLLM ---"
cat <<PYLAUNCH | colab exec -s "${SESSION}" --timeout 120
import os, subprocess, textwrap

# The PyPI vLLM wheel links against CUDA 13 (libcudart.so.13) while Colab's
# torch is cu128, so the CUDA 13 runtime that pip pulled in as a dependency has
# to be on the loader path or every vllm import dies with an ImportError.
script = textwrap.dedent('''\\
#!/usr/bin/env bash
set -euo pipefail
export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib:\${LD_LIBRARY_PATH:-}
export VLLM_LOGGING_LEVEL=INFO
exec vllm serve ${MODEL_ID} \\\\
  --served-model-name ${SERVED_NAME} ${MODEL_ID} \\\\
  --host 0.0.0.0 --port ${PORT} \\\\
  --dtype bfloat16 \\\\
  --max-model-len ${MAX_MODEL_LEN} \\\\
  --gpu-memory-utilization ${GPU_MEM_UTIL} \\\\
  --max-num-seqs ${MAX_NUM_SEQS} \\\\
  --enable-prefix-caching \\\\
  --reasoning-parser qwen3 \\\\
  --api-key "${API_KEY}" \\\\
  --trust-remote-code
''')
os.makedirs("/content/logs", exist_ok=True)
open("/content/serve_vllm.sh", "w").write(script)
os.chmod("/content/serve_vllm.sh", 0o755)

subprocess.run("pkill -f 'vllm serve' 2>/dev/null; sleep 5", shell=True)
# setsid puts the server in its own session, so a kernel restart — which kills
# the kernel's process group — no longer takes the server down with it. Plain
# nohup is not enough here; that was measured, not assumed.
subprocess.run("cd /content && setsid nohup ./serve_vllm.sh > /content/logs/vllm.log 2>&1 < /dev/null &", shell=True)
print(">>> launched", flush=True)
PYLAUNCH

# ── 4. Wait for readiness ────────────────────────────────
echo ""
echo "--- [4/5] Waiting for the model to load (cold start ~5-7 min) ---"
deadline=$(( $(date +%s) + READY_TIMEOUT ))
ready=0
while [ "$(date +%s)" -lt "${deadline}" ]; do
  # Each exec is short: the client's --timeout bounds total call time, not idle
  # time, so one long poll inside the VM would trip it.
  status_line=$(cat <<PYPOLL | colab exec -s "${SESSION}" --timeout 120 2>/dev/null | tail -1
import subprocess, time
for _ in range(10):
    code = subprocess.run(
        "curl -s -m 3 -o /dev/null -w '%{http_code}' "
        "-H 'Authorization: Bearer ${API_KEY}' http://127.0.0.1:${PORT}/v1/models",
        shell=True, capture_output=True, text=True).stdout.strip()
    alive = subprocess.run("pgrep -f 'vllm serve' >/dev/null", shell=True).returncode == 0
    if code == "200" or not alive:
        break
    time.sleep(9)
print(f"{code}:{alive}", flush=True)
PYPOLL
)
  case "${status_line}" in
    200:*) ready=1; break ;;
    *:False) echo "vLLM exited during startup." >&2; break ;;
    *) echo "    still loading (${status_line})" ;;
  esac
done

if [ "${ready}" != "1" ]; then
  echo "" >&2
  echo "❌ Server did not become ready within ${READY_TIMEOUT}s." >&2
  exit 1
fi
echo "✅ Server is ready."

# ── 5. Open the port ─────────────────────────────────────
echo ""
echo "--- [5/5] Opening a public tunnel ---"
PUBLIC_URL=$(cat <<PYTUNNEL | colab exec -s "${SESSION}" --timeout 180 2>/dev/null | tail -1
import os, re, subprocess, time

if not os.path.exists("/content/cloudflared"):
    subprocess.run(
        "curl -sSL -o /content/cloudflared "
        "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 "
        "&& chmod +x /content/cloudflared", shell=True)

# A quick tunnel needs no Cloudflare account and no interactive login, which is
# what makes it usable from a headless agent. Colab's own proxyPort() does not
# work here: it round-trips through the browser frontend and hangs the kernel.
subprocess.run("pkill -f 'cloudflared tunnel' 2>/dev/null; rm -f /content/logs/cf.log; sleep 2", shell=True)
subprocess.run("setsid nohup /content/cloudflared tunnel --no-autoupdate "
               "--url http://127.0.0.1:${PORT} > /content/logs/cf.log 2>&1 < /dev/null &", shell=True)

url = None
for _ in range(24):
    time.sleep(5)
    log = open("/content/logs/cf.log", errors="ignore").read()
    match = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", log)
    if match:
        url = match.group(0)
        break
print(url or "", flush=True)
PYTUNNEL
)

if [ -z "${PUBLIC_URL}" ]; then
  echo "❌ Tunnel did not produce a URL." >&2
  exit 1
fi

echo ""
echo "--- Verifying from this machine ---"
if curl -sf -m 30 -H "Authorization: Bearer ${API_KEY}" "${PUBLIC_URL}/v1/models" >/dev/null; then
  echo "✅ ${PUBLIC_URL}/v1/models answered."
else
  echo "⚠️  The endpoint did not answer from here yet; it may need a few more seconds." >&2
fi

cat <<EOF

═══════════════════════════════════════════════════════
🎉  Serving ${MODEL_ID}

    Base URL:  ${PUBLIC_URL}/v1
    API key:   ${API_KEY}
    Model:     ${SERVED_NAME}

    curl ${PUBLIC_URL}/v1/chat/completions \\
      -H "Authorization: Bearer ${API_KEY}" \\
      -H 'Content-Type: application/json' \\
      -d '{"model":"${SERVED_NAME}","messages":[{"role":"user","content":"Hello"}]}'

    Point the serving shim at it:
      export LEXI_BACKEND_URL=${PUBLIC_URL}/v1
      export LEXI_BACKEND_MODEL=${SERVED_NAME}
      export LEXI_BACKEND_API_KEY=${API_KEY}

    The VM bills compute units until you stop it:
      STOP=1 SESSION=${SESSION} bash $0
═══════════════════════════════════════════════════════
EOF
