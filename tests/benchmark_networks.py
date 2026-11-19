"""Benchmark separate and species-packed networks: python -m tests.benchmark_networks."""
import argparse
import statistics
import time
import torch
from abfml.builtin.networks import SpeciesFittingNetwork, ResidualMLP

def median_runtime(function, repeat: int) -> float:
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        times.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(times)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeat', type=int, default=15)
    parser.add_argument('--backward', action='store_true')
    parser.add_argument('--centers', type=int, default=128)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(20261030)
    network_size = [52, 64, 64]
    energy_shift = [0.1, -0.2, 0.3]
    packed = SpeciesFittingNetwork(3, network_size, 'tanh', True, True, energy_shift).to(torch.float64).eval()
    independent = [ResidualMLP(network_size=network_size + [1], activate='tanh', bias=True, resnet_dt=True, energy_shift=energy_shift[type_id], activate_output=False, shifted_output=True).to(torch.float64).eval() for type_id in range(3)]
    with torch.no_grad():
        for type_id, network in enumerate(independent):
            for packed_layer, single_layer in zip(packed.layers, network.linear):
                single_layer.weight.copy_(packed_layer.weight[type_id])
                single_layer.bias.copy_(packed_layer.bias[type_id])
            network.resnet[1, :64].copy_(packed.resnet[type_id, 1, :64])
    inputs = torch.randn(args.centers, 52, dtype=torch.float64)
    species = torch.arange(args.centers, dtype=torch.long) % 3

    def evaluate_packed() -> torch.Tensor:
        if args.backward:
            return packed(inputs, species).sum()
        with torch.no_grad():
            return packed(inputs, species).sum()

    def evaluate_independent() -> torch.Tensor:
        if args.backward:
            output = inputs.new_zeros(args.centers, 1)
            for type_id, network in enumerate(independent):
                mask = species == type_id
                output[mask] = network(inputs[mask])
            return output.sum()
        with torch.no_grad():
            output = inputs.new_zeros(args.centers, 1)
            for type_id, network in enumerate(independent):
                mask = species == type_id
                if bool(mask.any()):
                    output[mask] = network(inputs[mask])
            return output.sum()
    for name, function in (('independent', evaluate_independent), ('packed', evaluate_packed)):
        for _ in range(3):
            result = function()
            if args.backward:
                result.backward()
        elapsed = median_runtime(lambda: function().backward() if args.backward else function(), args.repeat)
        mode = 'forward+backward' if args.backward else 'forward'
        print(f'{name:12s} {mode:17s} median: {elapsed:9.3f} ms')
if __name__ == '__main__':
    main()
