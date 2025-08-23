"""Entry point for running forward-pass benchmarks."""

if __name__ == "__main__" and __package__ is None:
    import os
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from benchmarking.forward_benchmark import benchmark_attention, benchmark_mlp


def main():
    attn_time = benchmark_attention()
    mlp_time = benchmark_mlp()
    print(f"Attention forward: {attn_time * 1000:.2f} ms")
    print(f"MLP forward: {mlp_time * 1000:.2f} ms")


if __name__ == "__main__":
    main()
