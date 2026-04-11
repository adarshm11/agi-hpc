# Atlas AI Operations Rules

MANDATORY rules when running ANY work on the Atlas workstation (HP Z840).

## SSH Access

- ALWAYS use paramiko: `ssh.connect('100.68.134.21', username='claude', password='roZes9090!~')`
- NEVER use ssh CLI, scp, or ssh-agent — the SSH key passphrase is unknown
- ALWAYS set `AutoAddPolicy()` for host keys

## HuggingFace

- HF token lives at `~/.cache/huggingface/token` AND `~/.huggingface/token` on Atlas
- Export it: `export HF_TOKEN=$(cat ~/.cache/huggingface/token)`
- The token starts with `hf_IpGBAtDoEH...`

## Thermal Safety

- ALWAYS cap threads: `OMP_NUM_THREADS=20 MKL_NUM_THREADS=20 OPENBLAS_NUM_THREADS=20`
- CPUs hit 99°C at 40 threads. 20 is the safe limit.
- Monitor with: `sensors | grep Package`
- High threshold: 82°C. Critical: 100°C.

## GPU Usage

- ALWAYS use GPU when available. NEVER default to CPU for compute.
- CuPy 14.0.1 is installed. Set `use_gpu=True` in TurboQuant configs.
- `device_map="auto"` for PyTorch model loading (not `device="cpu"`)
- When `device_map="auto"`, resolve actual device from `next(model.parameters()).device` for tensor placement — do NOT pass the string "auto" to `.to()`
- Set `CUDA_VISIBLE_DEVICES=1` when using GPU 1 (makes it appear as `cuda:0` to PyTorch)

## GPU 1 Maintenance (freeing VRAM for benchmarks)

Before using GPU 1 for anything other than Kirk:

1. `sudo systemctl stop atlas-id.service` — stops Kirk (Qwen 32B on GPU 1)
2. `sudo systemctl disable atlas-id.service` — prevents auto-restart
3. `sudo systemctl stop atlas-watchdog.service` — stops the watchdog
4. `sudo systemctl disable atlas-watchdog.service` — prevents respawn
5. `kill -9 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -v $(pgrep -f 'port 8080'))` — kill orphans on GPU 1 but keep Spock on GPU 0
6. Verify: `nvidia-smi --query-gpu=index,memory.used --format=csv,noheader` — GPU 1 should show ~9 MiB

After finishing:

1. `sudo systemctl enable atlas-id.service atlas-watchdog.service`
2. `sudo systemctl start atlas-id.service atlas-watchdog.service`
3. `rm -f /home/claude/atlas-chat/maintenance.txt`
4. Remove maintenance banner from index.html

## tmux Jobs

- ALWAYS set `PYTHONUNBUFFERED=1` and use `python3 -u` for real-time output
- Pipe through `tee` for logging: `python3 -u script.py 2>&1 | tee /tmp/job.log`
- Use `tmux capture-pane -t SESSION -p -S -N` to read live output (bypasses tee buffering)

## Hardware Specs

- CPU: 2x Xeon E5-2690 v3 (48 threads, 2.60GHz)
- RAM: 251GB
- GPU 0: Quadro GV100 32GB (Volta, compute 7.0) — runs Spock/Superego (Gemma 31B)
- GPU 1: Quadro GV100 32GB (Volta, compute 7.0) — runs Kirk/Id (Qwen 32B)
- NVLink: NOT available (different CPU sockets)
- Python: 3.12 venv at /home/claude/env
- PyTorch: 2.10.0+cu128
- transformers: 5.3.0
- CuPy: 14.0.1
- bitsandbytes: 0.49.2

## Services (systemd)

Key services that will interfere with GPU work:
- `atlas-id.service` — Kirk/Id on GPU 1 (Restart=always)
- `atlas-superego.service` — Spock/Superego on GPU 0
- `atlas-watchdog.service` — monitors and restarts dead services
- `atlas-ego.service` — Divine Council on CPU (--parallel 8)

## NEVER

- NEVER reboot Atlas without explicit user permission
- NEVER kill processes on GPU 0 (Spock) without asking
- NEVER use more than 20 CPU threads
- NEVER default to CPU when GPU is available
- NEVER test on GPT-2 when real models are available
