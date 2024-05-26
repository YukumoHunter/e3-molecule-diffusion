import jax
import jax.numpy as jnp
from jax import device_put
from jax import random
from jax import jit
from jax.scipy.special import logsumexp
from jax.nn import softplus
from jax.nn.initializers import uniform, variance_scaling
from jax.nn.initializers import uniform, kaiming_uniform
import pickle

from jax import nn
from qm9.analyze import analyze_stability_for_molecules


def one_hot(indices, num_classes):
    output = jnp.zeros(indices.shape + (num_classes,))
    output = output.at[jnp.arange(indices.size), indices.ravel()].set(1)
    return output.reshape(indices.shape + (num_classes,))


def flatten_sample_dictionary(samples):
    results = {"one_hot": [], "x": [], "node_mask": []}
    for number_of_atoms in samples:
        positions = samples[number_of_atoms]["_positions"]
        atom_types = samples[number_of_atoms]["_atomic_numbers"]

        for positions_single_molecule, atom_types_single_molecule in zip(
            positions, atom_types
        ):
            mask = jnp.ones(positions.shape[1])

            one_hot = one_hot(indices, num_classes=10)
            nn.one_hot(jnp.array(atom_types_single_molecule), num_classes=10)

            results["x"].append(jnp.array(positions_single_molecule))
            results["one_hot"].append(jnp.array(one_hot))
            results["node_mask"].append(jnp.array(mask))

    return results


def main():
    with open("generated_samples/gschnet/gschnet_samples.pickle", "rb") as f:
        samples = pickle.load(f)

    from configs import datasets_config

    dataset_info = {
        "atom_decoder": [None, "H", None, None, None, None, "C", "N", "O", "F"],
        "name": "qm9",
    }

    results = flatten_sample_dictionary(samples)

    print(f'Analyzing {len(results["x"])} molecules...')

    validity_dict, rdkit_metrics = analyze_stability_for_molecules(
        results, dataset_info
    )
    print(validity_dict, rdkit_metrics[0])


if __name__ == "__main__":
    main()
