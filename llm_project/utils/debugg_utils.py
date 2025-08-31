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
