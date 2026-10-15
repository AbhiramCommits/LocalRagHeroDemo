#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# LocalRAG Hero Demo — GPU bring-up preflight.
#
# Ordered, idempotent checks that report PASS / FAIL / SKIP for the things that
# actually break a local-GPU RAG deployment: NVIDIA driver, driver/CUDA floor,
# the container toolkit + `docker run --gpus all`, WSL2, free disk, free VRAM
# and the Hugging Face cache location.
#
# Exit code is non-zero ONLY for hard failures (problems that break even the
# CPU-only path, e.g. no Python or almost no disk). GPU-specific gaps are SKIP
# with a remediation hint, because the demo still runs on CPU.
#
#   bash scripts/bringup_check.sh | tee results/bringup_report.txt
# ---------------------------------------------------------------------------
set -uo pipefail

PASS=0; FAIL=0; SKIP=0; HARD=0
MIN_DISK_GB="${MIN_DISK_GB:-10}"
MIN_VRAM_MB="${MIN_VRAM_MB:-4096}"
DRIVER_FLOOR="${DRIVER_FLOOR:-525}"

c_pass() { printf '[PASS] %s\n' "$1"; PASS=$((PASS + 1)); }
c_skip() { printf '[SKIP] %s\n' "$1"; [ -n "${2:-}" ] && printf '       remediation: %s\n' "$2"; SKIP=$((SKIP + 1)); }
c_fail() { printf '[FAIL] %s\n' "$1"; [ -n "${2:-}" ] && printf '       remediation: %s\n' "$2"; FAIL=$((FAIL + 1)); [ "${3:-soft}" = "hard" ] && HARD=$((HARD + 1)); }

echo "LocalRAG Hero Demo — GPU bring-up preflight"
echo "==========================================="
echo "host: $(uname -srm 2>/dev/null || echo unknown)   date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo

# --- 1. Python ------------------------------------------------------------- #
if command -v python3 >/dev/null 2>&1; then
    c_pass "python3 present: $(python3 --version 2>&1)"
else
    c_fail "python3 not found" "install Python 3.11 (conda create -n localrag python=3.11)" hard
fi

# --- 2. Free disk ---------------------------------------------------------- #
AVAIL_KB=$(df -Pk . 2>/dev/null | awk 'NR==2 {print $4}')
if [ -n "${AVAIL_KB:-}" ]; then
    AVAIL_GB=$((AVAIL_KB / 1024 / 1024))
    if [ "$AVAIL_GB" -ge "$MIN_DISK_GB" ]; then
        c_pass "free disk ${AVAIL_GB} GB (>= ${MIN_DISK_GB} GB)"
    elif [ "$AVAIL_GB" -ge 5 ]; then
        c_skip "free disk only ${AVAIL_GB} GB (< ${MIN_DISK_GB} GB recommended)" "free space before pulling model weights"
    else
        c_fail "free disk critically low: ${AVAIL_GB} GB" "free at least ${MIN_DISK_GB} GB" hard
    fi
else
    c_skip "could not determine free disk" "run df -h manually"
fi

# --- 3. NVIDIA driver ------------------------------------------------------ #
if command -v nvidia-smi >/dev/null 2>&1; then
    DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' ')
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1)
    c_pass "nvidia-smi present — driver ${DRIVER:-unknown}, GPU '${GPU_NAME:-unknown}'"

    # --- 3b. driver / CUDA version floor ---
    MAJOR=${DRIVER%%.*}
    if [ -n "${MAJOR:-}" ] && [ "$MAJOR" -ge "$DRIVER_FLOOR" ] 2>/dev/null; then
        c_pass "driver ${DRIVER} meets floor >= ${DRIVER_FLOOR}"
    else
        c_fail "driver ${DRIVER:-unknown} is below the recommended floor ${DRIVER_FLOOR}" \
               "install a newer NVIDIA driver (Linux: distribution driver or .run; Windows/WSL: the Windows NVIDIA driver)" soft
    fi
    CUDA_TXT=$(nvidia-smi 2>/dev/null | grep -o 'CUDA Version: [0-9.]*' | head -n1 | awk '{print $3}')
    if [ -n "${CUDA_TXT:-}" ]; then
        CUDA_MAJOR=${CUDA_TXT%%.*}
        if [ "${CUDA_MAJOR:-0}" -ge 12 ] 2>/dev/null; then
            c_pass "CUDA runtime ${CUDA_TXT} (>= 12.0)"
        else
            c_skip "CUDA runtime ${CUDA_TXT} (< 12.0)" "update the driver to expose CUDA >= 12.0"
        fi
    fi

    # --- 6. Free VRAM vs requirement ---
    FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d ' ')
    if [ -n "${FREE_MB:-}" ]; then
        if [ "$FREE_MB" -ge "$MIN_VRAM_MB" ]; then
            c_pass "free VRAM ${FREE_MB} MiB (>= ${MIN_VRAM_MB} MiB required)"
        else
            c_fail "free VRAM ${FREE_MB} MiB < ${MIN_VRAM_MB} MiB required" \
                   "free GPU memory (stop other processes) or use a larger GPU" soft
        fi
    fi
else
    c_skip "nvidia-smi not found — no NVIDIA GPU on this host (CPU path only)" \
           "install the NVIDIA driver: Linux 'sudo apt-get install -y nvidia-driver-535', Windows 'winget install Nvidia.GeForceExperience'"
fi

# --- 4. Container toolkit / GPU-in-Docker ---------------------------------- #
if command -v docker >/dev/null 2>&1; then
    c_pass "docker present: $(docker --version 2>/dev/null)"
    if command -v nvidia-ctk >/dev/null 2>&1 || [ -e /usr/bin/nvidia-container-runtime ]; then
        c_pass "nvidia-container-toolkit detected"
    else
        c_skip "nvidia-container-toolkit not detected" \
               "sudo apt-get install -y nvidia-container-toolkit && sudo nvidia-ctk runtime configure --runtime=docker"
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
        if docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
            c_pass "docker run --gpus all smoke test succeeded"
        else
            c_fail "docker run --gpus all failed" \
                   "install nvidia-container-toolkit and restart docker (sudo systemctl restart docker)" soft
        fi
    else
        c_skip "docker GPU smoke test skipped (no NVIDIA GPU)" ""
    fi
else
    c_skip "docker not found (native install path still works)" \
           "install Docker Engine + nvidia-container-toolkit for the container path"
fi

# --- 5. WSL2 detection ----------------------------------------------------- #
if [ -r /proc/version ] && grep -qi microsoft /proc/version 2>/dev/null; then
    c_pass "running under WSL2 ($(grep -o 'microsoft[^ ]*' /proc/version | head -n1))"
    c_skip "WSL: no NVIDIA driver is installed inside the distro (by design)" \
           "install the Windows NVIDIA driver on the host; /usr/lib/wsl/lib is injected automatically"
else
    c_pass "not running under WSL"
fi

# --- 7. Hugging Face cache ------------------------------------------------- #
HF_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
if mkdir -p "$HF_DIR" 2>/dev/null && [ -w "$HF_DIR" ]; then
    c_pass "Hugging Face cache writable at ${HF_DIR}"
else
    c_fail "Hugging Face cache not writable: ${HF_DIR}" \
           "export HF_HOME=/path/to/writable/dir and re-run" soft
fi

echo
echo "-------------------------------------------"
printf 'summary: %d PASS, %d FAIL, %d SKIP\n' "$PASS" "$FAIL" "$SKIP"
if [ "$HARD" -gt 0 ]; then
    echo "result: HARD FAILURE(S) — resolve the [FAIL] hard items above."
    exit 1
fi
echo "result: no hard failures. GPU-specific SKIPs mean those configs will be"
echo "        recorded as 'not_measured_on_this_host' in the benchmark table."
exit 0
