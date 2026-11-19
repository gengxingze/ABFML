"""CPU benchmark for ACSF angular descriptors: python -m tests.benchmark_acsf."""

import argparse
import statistics
import time

import torch

from abfml.builtin.acsf import ACSFAngularDescriptor, AngularBasis


def make_inputs(
    n_centers: int = 32,
    max_neighbors: int = 32,
    num_types: int = 3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(20261029)
    vectors = torch.randn(
        n_centers,
        max_neighbors,
        3,
        generator=generator,
        dtype=torch.float64,
    )
    neighbor_types = torch.randint(
        0,
        num_types,
        (n_centers, max_neighbors),
        generator=generator,
    )
    valid = torch.rand(
        n_centers,
        max_neighbors,
        generator=generator,
        dtype=torch.float64,
    ) > 0.1
    vectors = vectors * valid.unsqueeze(-1)
    return neighbor_types, vectors, valid


def make_descriptor(num_types: int, include_rjk: bool) -> ACSFAngularDescriptor:
    return ACSFAngularDescriptor(
        num_types=num_types, cutoff=5., include_rjk=include_rjk,
        basis=(AngularBasis(.01,0.,-1.,1), AngularBasis(.03,.5,1.,2),
               AngularBasis(.1,1.,-1.,4), AngularBasis(.3,1.5,1.,8)),
    ).double().eval()


def median_runtime(function, repeat: int) -> float:
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        times.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(times)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--backward", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)

    for include_rjk in (False, True):
        neighbor_types, vectors, valid = make_inputs()
        descriptor = make_descriptor(3, include_rjk)
        if args.backward:
            vectors.requires_grad_(True)

        def evaluate() -> torch.Tensor:
            if args.backward:
                if vectors.grad is not None:
                    vectors.grad = None
                output = descriptor(neighbor_types, vectors, valid)
                output.square().sum().backward()
                return output
            with torch.no_grad():
                return descriptor(neighbor_types, vectors, valid)

        for _ in range(2):
            evaluate()
        elapsed = median_runtime(evaluate, args.repeat)
        name = "angular_narrow" if include_rjk else "angular_wide"
        mode = "forward+backward" if args.backward else "forward"
        print(f"{name:10s} {mode:17s} median: {elapsed:9.3f} ms")


if __name__ == "__main__":
    main()
