import time
import os
import platform
import psutil


class Colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


# Process/GPU memory helpers


def get_proc_mem_mb():
    """
    Returns (rss_mb, uss_mb_or_None) for the current Python process.
    - RSS: resident set size (what the OS keeps in RAM for this process)
    - USS: unique set size (memory private to this process) if available
    """
    import os
    import psutil

    p = psutil.Process(os.getpid())
    try:
        m = p.memory_full_info()  # may include 'uss' on many platforms
        rss_mb = m.rss / (1024**2)
        uss = getattr(m, "uss", None)
        uss_mb = (uss / (1024**2)) if uss is not None else None
    except Exception:
        m = p.memory_info()
        rss_mb = m.rss / (1024**2)
        uss_mb = None
    return rss_mb, uss_mb


def get_gpu_mem_mb():
    """
    Returns (torch_alloc_mb, torch_reserved_mb, nvml_proc_used_mb_or_None)
    - torch_alloc: memory occupied by tensors
    - torch_reserved: memory reserved by PyTorch caching allocator
    - nvml_proc_used: total GPU bytes attributed to this PID via NVML (if available)
    """
    torch_alloc = torch_reserved = nvml_proc = None
    try:
        import torch

        if torch.cuda.is_available():
            torch_alloc = torch.cuda.memory_allocated() / (1024**2)
            torch_reserved = torch.cuda.memory_reserved() / (1024**2)
    except Exception:
        pass

    try:
        import os
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        procs = pynvml.nvmlDeviceGetComputeRunningProcesses_v2(handle)
        this_pid = os.getpid()
        for pr in procs:
            if pr.pid == this_pid:
                nvml_proc = pr.usedGpuMemory / (1024**2)  # MB
                break
    except Exception:
        nvml_proc = None
    return torch_alloc, torch_reserved, nvml_proc


def print_mem_line(prefix=""):
    """
    Pretty one-liner you can call anywhere.
    Example: print_mem_line(prefix=f\"[step {it}]\")
    """
    rss_mb, uss_mb = get_proc_mem_mb()
    ga, gr, nvmlp = get_gpu_mem_mb()
    parts = [prefix.strip(), f"RSS {rss_mb:.2f}MB"]
    if uss_mb is not None:
        parts.append(f"USS {uss_mb:.2f}MB")
    if ga is not None:
        parts.append(f"CUDA alloc {ga:.2f}MB")
    if gr is not None:
        parts.append(f"CUDA reserved {gr:.2f}MB")
    if nvmlp is not None:
        parts.append(f"GPU(proc) {nvmlp:.2f}MB")
    print(" | ".join(parts))


# CPU util
_PROC = psutil.Process(os.getpid())
_CPU_COUNT = max(psutil.cpu_count(logical=True) or 1, 1)


def get_proc_cpu_percent(
    prime: bool = False,
    interval: float = 0.0,
    normalized: bool = True,
    cap: bool = True,
) -> float:
    """
    Per-process CPU% (this Python process).
    - First call should be prime=True (returns 0.0) to start the measurement window.
    - Subsequent calls (prime=False) return % CPU used since the previous call.
    Note: On multi-core, this can exceed 100% (e.g., 200% on 2 full cores).
    """
    if prime:
        _PROC.cpu_percent(None)  # start baseline
        return 0.0
    val = _PROC.cpu_percent(interval=interval)  # can be up to 100 * num_cores
    if normalized:
        val = val / _CPU_COUNT  # -> 0–100% scale
        if cap:
            val = min(val, 100.0)
    return val


def print_resource_usage(self, step: int):
    now = time.time()
    step_duration = now - self._last_step_time
    self._last_step_time = now

    ram_used = psutil.virtual_memory().used / 1024**2
    cpu_p = psutil.cpu_percent()

    print(
        f"""
            [DEBUG]{f" BPE Merge step: {step:>10}m":>25}||{f"{'Step duration:':<8}{step_duration:>8.2f}s":>24}||
            {f"        Ram used: {ram_used:>15.2f}MB":>32}||{f"{'CPU:':>1} {cpu_p:>17.2f}%":>24}||"""
    )
    current_os = platform.system().lower()

    if current_os == "linux":
        if os.path.exists("/kaggle/working"):
            path = "/kaggle/working"
        elif os.path.exists("/content"):
            path = "/content"
        else:
            path = "/"
        print(f"             Disk Usage ({path}): {psutil.disk_usage(path).percent}%")

    elif current_os == "windows":
        root_drive = os.getenv("SystemDrive", "C:") + "\\"
        print(
            f"             Disk Usage ({root_drive}): {psutil.disk_usage(root_drive).percent}%"
        )

    elif current_os == "darwin":
        print(f"             Disk Usage (/): {psutil.disk_usage('/').percent}%")

    else:
        print(f"{Colors.WARNING}[WARN]{Colors.ENDC} Unknown OS, defaulting to '/'")
        print(f"             Disk Usage (/): {psutil.disk_usage('/').percent}%")

    return step_duration
