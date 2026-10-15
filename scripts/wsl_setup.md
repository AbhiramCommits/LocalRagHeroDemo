# WSL2 setup for the LocalRAG Hero Demo

Run the demo on a Windows workstation with an NVIDIA GPU by using WSL2. The rule
that trips people up: **you install the NVIDIA driver on Windows, never inside
the WSL distro.** WSL injects `nvidia-smi` and the CUDA libraries automatically.

## 1. Install WSL2 (Windows 11 / Windows 10 21H2+)

Open **PowerShell as Administrator**:

```powershell
wsl --install -d Ubuntu-22.04
wsl --set-default-version 2
wsl --update
```

Reboot, then launch "Ubuntu" from the Start menu and create your Linux user.

Verify:

```powershell
wsl -l -v          # STATE should be Running, VERSION 2
```

## 2. Install the NVIDIA Windows driver (do NOT install a driver in WSL)

1. Download the **Windows** GeForce/Quadro/Data-Center driver from
   <https://www.nvidia.com/Download/index.aspx> (or `winget install Nvidia.GeForceExperience`).
2. Install it on the Windows host and reboot.
3. This installs the WSL GPU paravirtualisation support. There is **no**
   `nvidia-driver-*` apt package to install inside Ubuntu — doing so breaks the demo.

Verify inside Ubuntu:

```bash
nvidia-smi        # should print your GPU, driver version and CUDA version
ls /usr/lib/wsl/lib
```

If `nvidia-smi` is missing inside WSL but works in Windows, run `wsl --update`
and reboot. If it still fails, confirm you did **not** install a Linux driver.

## 3. CUDA toolkit inside the distro (only needed to compile extensions)

The runtime that ships through WSL is enough to run PyTorch. Install the toolkit
only if a package needs `nvcc`:

```bash
sudo apt-get update
sudo apt-get install -y build-essential
# Optional (matches the driver's CUDA major):
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update && sudo apt-get install -y cuda-toolkit-12-4
```

## 4. Docker Desktop + WSL integration

1. Install Docker Desktop for Windows and enable
   **Settings → Resources → WSL integration → Ubuntu-22.04**.
2. Ensure your Windows user is in the `docker-users` group.
3. Verify GPU access from inside WSL:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

If that fails, update Docker Desktop and the NVIDIA Windows driver.

## 5. Run the demo

```bash
git clone <your-repo-url> LocalRagHeroDemo && cd LocalRagHeroDemo
bash scripts/bringup_check.sh | tee results/bringup_report.txt   # step 0
conda env create -f environment.yml && conda activate localrag
pip install -e ".[all]"
make corpus index demo
# or, containerised:
docker compose run --rm localrag demo
```

## Troubleshooting quick table

| Symptom | Cause | Fix |
| --- | --- | --- |
| `nvidia-smi: command not found` in WSL | No Windows driver, or WSL not updated | Install the Windows NVIDIA driver, `wsl --update`, reboot |
| `nvidia-smi` works in Windows, not WSL | A Linux driver was installed inside the distro | `sudo apt-get purge 'nvidia-*'`, reboot WSL (`wsl --shutdown`) |
| `docker run --gpus all` → "could not select device driver" | nvidia-container-toolkit / WSL integration missing | Enable Docker Desktop WSL integration, update Docker |
| Very slow generation | Model on CPU because CUDA not visible to torch | `python -c "import torch; print(torch.cuda.is_available())"` must be `True` |
| Out-of-memory on GPU | Model + context too large for the GPU | Use a smaller model or the vLLM/NIM endpoint path |
