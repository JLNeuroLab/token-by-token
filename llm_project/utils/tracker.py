import time
from functools import wraps
from llm_project.utils.debugg_utils import Colors  # adjust path if needed


def track(func=None, v=False):
    def deco(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            name = f"{Colors.BOLD}{f.__name__}(){Colors.ENDC}"
            if v:
                print(f"{name} {Colors.OKBLUE}[STARTING]{Colors.ENDC}")
            start = time.time()
            result = f(*args, **kwargs)
            duration = time.time() - start
            print(f"{name} {Colors.OKGREEN}[DONE]{Colors.ENDC} {duration:.2f}s")
            return result

        return wrapper

    return deco(func) if func is not None else deco
