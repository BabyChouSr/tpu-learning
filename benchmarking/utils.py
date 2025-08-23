import time
import functools
import jax


def _block_until_ready(x):
    arr = getattr(x, "array", x)
    return jax.block_until_ready(arr)


def time_forward(fn, *args, warmup=5, iters=20):
    """Time the average JIT-compiled forward pass of `fn`.

    Args:
        fn: Callable to execute.
        *args: Arguments forwarded to `fn`.
        warmup: Number of warmup iterations before timing.
        iters: Number of timed iterations.
    """
    jit_fn = jax.jit(fn)
    for _ in range(warmup):
        _block_until_ready(jit_fn(*args))
    start = time.perf_counter()
    for _ in range(iters):
        _block_until_ready(jit_fn(*args))
    end = time.perf_counter()
    return (end - start) / iters


def trace_function(logdir, create_perfetto_link=False):
    """Decorator to profile a function with ``jax.profiler.trace``.

    Args:
        logdir: Directory where the trace file will be written.
        create_perfetto_link: Whether to create a Perfetto link (see JAX docs).
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with jax.profiler.trace(logdir, create_perfetto_link=create_perfetto_link):
                return fn(*args, **kwargs)
        return wrapper

    return decorator
