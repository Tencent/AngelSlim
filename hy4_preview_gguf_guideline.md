## What this preview does

It runs one large Hy4 Preivew GGUF model across two Linux machines:

- The first machine uses four RTX A4000 16 GB cards. One `prima-server` process controls all four cards and provides the API.
- The second machine uses one RTX 4090 Laptop GPU 16 GB and runs the other `prima-server` process.

> This is an advanced preview. The current public open-source edition does not include this MoE support.

## What you need

- Two x86-64 Linux machines with the GPUs listed above.
- A working NVIDIA driver on both machines.
- CUDA 12 and cuBLAS on the A4000 machine. The preview was built on Ubuntu 24.04 with CUDA 12.x.
- CUDA 13 and cuBLAS on the 4090 machine. The preview was built on Ubuntu 22.04 with CUDA 13.x.
- About 31 GB of system memory on each machine; more memory can improve file caching.
- Fast local storage on both machines.
- [Hy4-preview-STQ1_0.gguf](https://huggingface.co/AngelSlim/Hy4-preview-GGUF/tree/main) model files on both machines.

## Which asset to use

Most users should download these two runtime packages at [here](https://huggingface.co/AngelSlim/Hy4-preview-GGUF/tree/main/prima-runtime):

| Asset | Put it on | What it contains |
| --- | --- | --- |
| `prima-server-preview-a4000-rank0-linux-x86_64-cuda-runtime.tar.gz` | The machine with four A4000 cards | The rank-0 API server, the model-receipt and plan tools, and matching compiled libraries |
| `prima-server-preview-4090-rank1-linux-x86_64-cuda-runtime.tar.gz` | The machine with the 4090 Laptop GPU | The rank-1 worker, the model-receipt tool, and matching compiled libraries |

Keep each package's `bin/` and `lib/` directories together. The packaged server finds its libraries there automatically.

`SHA256SUMS` verifies the two runtime packages. For example:

```bash
sha256sum -c SHA256SUMS --ignore-missing
```

The manually uploaded runtime packages contain only compiled ELF executables, compiled shared libraries, and library symlinks. They do not contain source code, shell scripts, configuration files, model files, the NVIDIA driver, CUDA, or cuBLAS. GitHub's automatically generated “Source code” links are not runtime packages.

## Download and unpack

On the A4000 machine:

```bash
tar -xzf prima-server-preview-a4000-rank0-linux-x86_64-cuda-runtime.tar.gz
cd prima-server-preview-a4000-rank0-linux-x86_64-cuda-runtime
ldd ./bin/prima-server | sed -n '/not found/p'
```

On the 4090 machine:

```bash
tar -xzf prima-server-preview-4090-rank1-linux-x86_64-cuda-runtime.tar.gz
cd prima-server-preview-4090-rank1-linux-x86_64-cuda-runtime
ldd ./bin/prima-server | sed -n '/not found/p'
```

If the final command prints nothing, the server's shared-library dependencies were found. If it prints `not found`, install the named system or CUDA library before continuing.


</details>

## Prepare the two rank files

This preview deliberately does not ship a receipt or manual plan made on our machines. Those files bind the user's own model directory and GPU topology and must be generated locally.

1. On each machine, use `bin/prima-moe-artifact` to validate the six model files and create that machine's receipt:

```bash
./bin/prima-moe-artifact create-v1 \
  --adapter-id hyv4-middle-iq1-m-r8-v1 \
  --repository HongHuang/middle \
  --revision fc5f23d19a0d6552da1f1f3e39725d068c08444e \
  --manifest /path/to/model-manifest.json \
  --model-root /path/to/model-directory \
  --output rank0-receipt.prar
```

Use `--output rank1-receipt.prar` on the 4090 machine. The manifest must list the same repository revision plus the exact six file names, byte sizes, and SHA-256 values.

2. Capture the local GPU topology on each machine:

```bash
# A4000 machine
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  ./bin/prima-server --moe-device-group CUDA0,CUDA1,CUDA2,CUDA3 \
  --moe-topology-probe > rank0-topology.bin

# 4090 machine
CUDA_VISIBLE_DEVICES=0 \
  ./bin/prima-server --moe-device-group CUDA0 \
  --moe-topology-probe > rank1-topology.bin
```

3. Copy the rank-1 receipt and topology file to the A4000 machine, then create one shared plan there:

```bash
./bin/prima-moe-plan create-v1 \
  --adapter hyv4-middle-iq1-m-r8-v1 \
  --rank0-receipt rank0-receipt.prar \
  --rank1-receipt rank1-receipt.prar \
  --rank0-topology rank0-topology.bin \
  --rank1-topology rank1-topology.bin \
  --cut 61 \
  --rank0-device-layers 15,15,15,16 \
  --rank1-device-layers 17 \
  --output manual-plan.bin
```

Copy `manual-plan.bin` to the 4090 machine. Each machine should keep its own receipt, a copy of the same plan, and its own writable profile-cache path.

## Start the two servers

Use one data port and one signal port per rank. The example below uses direct host-to-host addresses; setting up cross-network connectivity is outside this guide.

In each unpacked runtime directory, create writable runtime directories and set the common Release configuration:

```bash
mkdir -p run/tmp run/logs
unset GGML_CUDA_P2P
export TMPDIR="$PWD/run/tmp"
export PRIMA_LOG_ROOT="$PWD/run/logs"
export CUDA_CACHE_DISABLE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_LAUNCH_BLOCKING=0
export GGML_CUDA_DISABLE_GRAPHS=1
export PRIMA_ACTIVATION_WIRE_DTYPE=fp32
export PRIMA_MOE_REQUEST_PLANNER_V1=1
export PRIMA_MOE_BG_PROMOTION_V1=0
export PRIMA_MOE_HOTCACHE_POLICY_V1=recent
export PRIMA_MOE_DECODE_TIMING_V1=1
export PRIMA_MOE_BACKGROUND_STAGING_V1=1
export PRIMA_MOE_PUBLISH_READY_BEFORE_DECODE_V1=1
export PRIMA_MOE_PRECISE_PREFETCH_V1=1
export PRIMA_MOE_ROUTE_PAGE_PREFETCH_V1=1
export PRIMA_MOE_ROUTE_PAGECACHE_POPULATE_V1=1
export PRIMA_MOE_STARTUP_COLD_PREFETCH_V1=1
export PRIMA_MOE_COPY_STREAM_V1=1
export PRIMA_MOE_COPY_STREAM_READY_SYNC_V1=1
export PRIMA_MOE_DRAIN_PROMOTION_COPIES_WITHIN_ALLOWANCE_V1=1
```

Start rank 1 on the 4090 machine first:

```bash
export CUDA_VISIBLE_DEVICES=0

./bin/prima-server \
  --model /path/to/model-directory/Hy4-preview-STQ1_0.gguf \
  --load-mode mmap --log-profile production \
  --topology static --world-timeout 7200 \
  --ctx-size 8192 --batch-size 2 --ubatch-size 1 --parallel 1 --n-predict -1 \
  --moe-hybrid manual \
  --moe-provider tiered-local-cold-hot-v1 \
  --moe-model-adapter hyv4-middle-iq1-m-r8-v1 \
  --moe-routing-policy local-cold-no-remote-v1 \
  --moe-min-request-hits 2 --moe-ttft-guard-percent 3 \
  --moe-artifact-receipt /path/to/rank1-receipt.prar \
  --moe-profile-cache /path/to/rank1-profile-cache.bin \
  --moe-manual-plan /path/to/manual-plan.bin \
  --jinja --no-context-shift --no-warmup --no-repack \
  --cache-prompt --cache-reuse 0 --no-slots --no-webui \
  --seed 0 --temp 0 --top-k 1 --top-p 1 --min-p 0 --repeat-penalty 1 \
  --next A4000_HOST:19867:20894 \
  --data-port 19868 --signal-port 20895 --worker-ready-listener \
  --host 127.0.0.1 --port 18728 \
  --device CUDA0 --split-mode layer --moe-device-group CUDA0 \
  --chat-template-kwargs '{"reasoning_effort":"no_think"}'
```

Then start rank 0 on the four-A4000 machine:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

./bin/prima-server \
  --model /path/to/model-directory/Hy4-preview-STQ1_0.gguf \
  --load-mode mmap --log-profile production \
  --topology static --head --world 2 --world-timeout 7200 \
  --ctx-size 8192 --batch-size 2 --ubatch-size 1 --parallel 1 --n-predict -1 \
  --moe-hybrid manual \
  --moe-provider tiered-local-cold-hot-v1 \
  --moe-model-adapter hyv4-middle-iq1-m-r8-v1 \
  --moe-routing-policy local-cold-no-remote-v1 \
  --moe-min-request-hits 2 --moe-ttft-guard-percent 3 \
  --moe-artifact-receipt /path/to/rank0-receipt.prar \
  --moe-profile-cache /path/to/rank0-profile-cache.bin \
  --moe-manual-plan /path/to/manual-plan.bin \
  --jinja --no-context-shift --no-warmup --no-repack \
  --cache-prompt --cache-reuse 0 --no-slots --no-webui \
  --seed 0 --temp 0 --top-k 1 --top-p 1 --min-p 0 --repeat-penalty 1 \
  --next RANK1_HOST:19868:20895 \
  --data-port 19867 --signal-port 20894 \
  --host 0.0.0.0 --port 18727 \
  --device CUDA0 --split-mode layer \
  --moe-device-group CUDA0,CUDA1,CUDA2,CUDA3 \
  --chat-template-kwargs '{"reasoning_effort":"no_think"}'
```

Replace `A4000_HOST`, `RANK1_HOST`, and every `/path/to/...` value. Allow the four Ring ports between the two machines. Only rank 0 provides the client API.

Wait until rank 0 reports healthy, then send one warmup request. OpenAI-compatible requests go to:

```text
http://A4000_HOST:18727/v1/chat/completions
```

The example deployment uses an 8K context. The server itself does not impose a generation-token limit, although a client may set `max_tokens` for an individual request.

## Expected performance

On the reference hardware, after one warmup and with one request running at a time, steady decode is approximately **1.02 seconds per token**. This is a reference result, not a guarantee: prompt length, model routing, storage speed, available memory, and other machine load can change it.

Prompt caching mainly improves the time before the first token. It does not by itself make later tokens decode faster.
