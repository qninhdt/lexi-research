# Serving Qwen3.5-4B on a Colab L4 with vLLM

`run_colab_serve.sh` rents a Colab L4, installs vLLM, serves the model, and
publishes an OpenAI-compatible endpoint over a Cloudflare quick tunnel.

```bash
bash run_colab_serve.sh
# ... prints a public base URL and a bearer token

STOP=1 bash run_colab_serve.sh   # release the VM when done
```

Knobs: `GPU`, `SESSION`, `MODEL_ID`, `SERVED_NAME`, `PORT`, `MAX_MODEL_LEN`,
`MAX_NUM_SEQS`, `GPU_MEM_UTIL`, `VLLM_VERSION`, `API_KEY`, `READY_TIMEOUT`.

The endpoint is a drop-in backend for the serving shim:

```bash
export LEXI_BACKEND_URL=https://<tunnel>.trycloudflare.com/v1
export LEXI_BACKEND_MODEL=lexi
export LEXI_BACKEND_API_KEY=<printed key>
```

Verified against the real `serve.service.grade` path — the base model validates
clean on the first attempt, no retries:

```
'The medicine helped ease his pain.'  meaning=4 grammar=4 naturalness=4 retries=0
'He ease the pain quick.'             meaning=4 grammar=0 naturalness=4 retries=0
  correction='He [ease>eases:agr] the pain [quick>quickly:form].'
```

## Measured on an L4 (22 GiB, sm89, driver 580.82, Python 3.12)

| | |
|---|---|
| Cold start, first session | ~7 min (install ~4 min, engine init 263 s of which 49 s is compile) |
| Warm start, compile cache hot | 161 s engine init |
| Weights + activations | 19.0 GiB of 22.0 GiB at `--gpu-memory-utilization 0.90` |
| KV cache | 9.3 GiB — 273k tokens paged, ~8.7x concurrency at full 32k context |

Throughput, 128-token completions, thinking disabled:

| Concurrency | Wall | Output tokens | Throughput | Avg latency |
|---|---|---|---|---|
| 1 | 1.15 s | 32 | 27.8 tok/s | 1.15 s |
| 8 | 1.86 s | 261 | 140.5 tok/s | 1.37 s |
| 32 | 2.67 s | 1041 | 389.8 tok/s | 1.86 s |

## Four things that are not obvious

**The PyPI vLLM wheel wants CUDA 13; Colab's torch is cu128.** Importing vLLM
fails with `ImportError: libcudart.so.13`. The CUDA 13 runtime *is* installed —
pip pulls it in as a vLLM dependency — just not on the loader path. The launcher
exports:

```bash
LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
```

Downgrading torch to match the wheel is the alternative, and it is worse: it
would rebuild the environment the training runs are pinned against.

**vLLM must be at least 0.26.0.** Qwen3.5-4B is
`Qwen3_5ForConditionalGeneration` — multimodal, with a vision tower and hybrid
linear/full attention (`linear_attention` and `full_attention` layers, Gated
DeltaNet). That architecture entered the vLLM registry in 0.26.0; earlier wheels
reject the checkpoint outright. vLLM selects the Triton/FLA GDN prefill kernel
for it and pads the mamba page size to match the attention page size.

**Background processes need `setsid`, not `nohup`.** `colab restart-kernel`
kills the kernel's whole process group, and a plain `nohup` server dies with it
— measured, not assumed. Launched under `setsid` (its own session id), both the
vLLM server and the tunnel survive a kernel restart, and the public endpoint
keeps answering throughout.

**Poll in short calls, never one long `colab exec`.** `--timeout` bounds the
whole call, so a multi-minute wait inside a single exec ends in
`TimeoutError: Timeout waiting for output` or `RuntimeError: Connection was
lost.` The work itself survives — it is detached — but the client's view of it
does not. Both the install and the readiness wait start work detached, then poll
from the local shell in ~90 s calls.

**An unquoted heredoc expands `$VAR` locally, before the VM ever sees it.** The
launcher builds the remote start script inside `cat <<PYLAUNCH`, which is
deliberately unquoted so `${MODEL_ID}` and friends interpolate. That also means
any `$` meant for the *remote* shell has to survive one round of local
expansion: `\${LD_LIBRARY_PATH:-}` reaches the VM intact, while
`\\${LD_LIBRARY_PATH:-}` expands here and ships this machine's paths to the VM.

That mistake is not loud. It wrote a literal `\/usr/local/cuda-12.5/lib64` — a
local path, with a stray backslash — into the remote `LD_LIBRARY_PATH`, which
broke CUDA library resolution for the server process only. `nvidia-smi` and
`torch.cuda.is_available()` both still reported a healthy L4; vLLM failed
several minutes later with `0 active driver(s) found (expected 1)` and
`RuntimeError: Failed to infer device type`. Read `/content/serve_vllm.sh` on
the VM when the server dies at startup but the GPU looks fine.


## Opening the port

Colab's built-in `google.colab.output.eval_js('google.colab.kernel.proxyPort(…)')`
round-trips through the browser frontend. From a headless CLI session there is
no frontend to answer, so the call **hangs the kernel** and needs
`colab restart-kernel` to recover. Cloudflare quick tunnels need no account and
no interactive login, which is what makes them usable here. `--url` prints an
`https://<random>.trycloudflare.com` hostname; the launcher greps it out of the
tunnel log.

That URL is public. The launcher generates a random `API_KEY` by default and
passes it to `vllm serve --api-key`, so unauthenticated requests get a 401.

## Reasoning output

The chat template emits `<think>` blocks, so a plain request returns the model's
reasoning inside `content`. `--reasoning-parser qwen3` splits it into a separate
`reasoning` field, leaving `content` as the answer alone:

```json
{"reasoning": "Thinking Process:\n\n1. **Analyze the Request:** ...",
 "content": "\n\n391"}
```

Per-request, `"chat_template_kwargs": {"enable_thinking": false}` skips thinking
altogether — worth it for structured grading calls, where the reasoning is spent
tokens.

Prefix caching works but vLLM labels it experimental for this model: enabling it
puts the Mamba cache in `align` mode, which it reports as experimental for Mamba
layers.

## Cleaning up

Each session's proxy token expires after an hour. When it does, the CLI prunes
its local record and the VM keeps running — and keeps billing. It then appears
in `colab sessions` as `[?] <endpoint>` with no name, and `colab stop` cannot
release it, because stop resolves by name and the name is what was pruned.
`colab new` does not reattach either; it allocates a second VM.

```bash
bash ops/release-colab-orphans.sh         # list orphans
bash ops/release-colab-orphans.sh --yes   # release them
```

Do not hand-edit `~/.config/colab-cli/sessions.json` to fake a record for an
orphan. It corrupts the live entry and turns one working session into a second
orphan.
