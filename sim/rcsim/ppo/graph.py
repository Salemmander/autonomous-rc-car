from collections.abc import Callable
from typing import Generic, TypeVar

import torch

T = TypeVar("T")


class Graphed(Generic[T]):
    """Run fn() through a CUDA graph: record it once, then replay with one launch.

    The networks here are tiny, so without graphs the time goes into Python
    launching thousands of small GPU kernels one by one, not into actual math.

    The first `warmup` calls run fn() normally (they still do real work). That
    lets PyTorch finish its lazy setup (cuBLAS handles, optimizer state, grads)
    before recording. Then one call records fn() and every call after replays it.

    Rules for fn, because a replay re-runs the recorded kernels on the same memory:
    - no Python-side decisions based on GPU values (no .item(), no `if tensor`)
    - inputs must live in the same tensors every call (copy new data into them)
    - outputs come back in the same tensors every call (read them after replay)
    """

    def __init__(
        self,
        fn: Callable[[], T],
        warmup: int = 3,
        stream: torch.cuda.Stream | None = None,
    ):
        self.fn = fn
        self.warmup = warmup
        # Recording happens on a side stream, as PyTorch requires. The env passes
        # its Warp stream so the env kernels end up in the recording too
        self.stream = stream or torch.cuda.Stream()
        self.calls = 0
        self.graph: torch.cuda.CUDAGraph | None = None
        self.outputs: T | None = None

    def __call__(self) -> T:
        if self.graph is not None:
            self.graph.replay()
            return self.outputs  # pyright: ignore[reportReturnType]  (set when recorded)

        current = torch.cuda.current_stream()
        self.stream.wait_stream(current)
        with torch.cuda.stream(self.stream):
            if self.calls < self.warmup:
                self.outputs = self.fn()
            else:
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph, stream=self.stream):
                    self.outputs = self.fn()
        current.wait_stream(self.stream)

        if self.graph is not None:
            # Recording doesn't run anything; replay to actually do this call's work
            self.graph.replay()
        self.calls += 1
        return self.outputs  # pyright: ignore[reportReturnType]
