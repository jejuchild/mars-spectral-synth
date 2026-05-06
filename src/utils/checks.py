import os
import time


def assert_cpu_only() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") not in ("", "-1"):
        raise RuntimeError("CUDA_VISIBLE_DEVICES is set; Phase 0 prototype is CPU-only.")
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is True; refusing to run on GPU.")


class RuntimeBudget:
    def __init__(self, seconds: float):
        self.seconds = float(seconds)
        self.start = time.perf_counter()

    def check(self) -> None:
        elapsed = time.perf_counter() - self.start
        if elapsed > self.seconds:
            raise RuntimeError(f"Runtime budget exceeded: {elapsed:.1f}s > {self.seconds:.1f}s")

    def remaining(self) -> float:
        return self.seconds - (time.perf_counter() - self.start)
