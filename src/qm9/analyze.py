try:
    from rdkit import Chem
    from qm9.rdkit_functions import BasicMolecularMetrics

    use_rdkit = True
except ModuleNotFoundError:
    use_rdkit = False
import qm9.dataset as dataset
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.stats as sp_stats
from qm9 import bond_analyze

import jax
import jax.numpy as jnp
from jax import device_put
from jax import random
from jax import jit
from jax.scipy.special import logsumexp
from jax.nn import softplus
from jax.nn.initializers import uniform, variance_scaling
from jax.nn.initializers import uniform, kaiming_uniform
from jax.scipy import stats as jsp_stats

# 'atom_decoder': ['H', 'B', 'C', 'N', 'O', 'F', 'Al', 'Si', 'P', 'S', 'Cl', 'As', 'Br', 'I', 'Hg', 'Bi'],

analyzed_19 = {
    "atom_types": {1: 93818, 3: 21212, 0: 139496, 2: 8251, 4: 26},
    "distances": [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        22566,
        258690,
        16534,
        50256,
        181302,
        19676,
        122590,
        23874,
        54834,
        309290,
        205426,
        172004,
        229940,
        193180,
        193058,
        161294,
        178292,
        152184,
        157242,
        189186,
        150298,
        125750,
        147020,
        127574,
        133654,
        142696,
        125906,
        98168,
        95340,
        88632,
        80694,
        71750,
        64466,
        55740,
        44570,
        42850,
        36084,
        29310,
        27268,
        23696,
        20254,
        17112,
        14130,
        12220,
        10660,
        9112,
        7640,
        6378,
        5350,
        4384,
        3650,
        2840,
        2362,
        2050,
        1662,
        1414,
        1216,
        966,
        856,
        492,
        516,
        420,
        326,
        388,
        326,
        236,
        140,
        130,
        92,
        62,
        52,
        78,
        56,
        24,
        8,
        10,
        12,
        18,
        2,
        10,
        4,
        2,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ],
}


class Histogram_discrete:
    def __init__(self, name="histogram"):
        self.name = name
        self.bins = {}

    def add(self, elements):
        for e in elements:
            if e in self.bins:
                self.bins[e] += 1
            else:
                self.bins[e] = 1

    def normalize(self):
        total = sum(self.bins.values())
        for key in self.bins:
            self.bins[key] = self.bins[key] / total

    def plot(self, save_path=None):
        width = 1  # the width of the bars
        x, y = [], []
        for key in self.bins:
            x.append(key)
            y.append(self.bins[key])

        plt.bar(x, y, width)
        plt.title(self.name)
        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()


class Histogram_cont:
    def __init__(
        self, num_bins=100, range=(0.0, 13.0), name="histogram", ignore_zeros=False
    ):
        self.name = name
        self.bins = jnp.zeros(num_bins)
        self.range = range
        self.ignore_zeros = ignore_zeros

    def add(self, elements):
        for e in elements:
            if not self.ignore_zeros or e > 1e-8:
                i = int(float(e) / self.range[1] * len(self.bins))
                i = min(i, len(self.bins) - 1)
                self.bins = jax.ops.index_add(self.bins, i, 1)

    def plot(self, save_path=None):
        width = (self.range[1] - self.range[0]) / len(
            self.bins
        )  # the width of the bars
        fig, ax = plt.subplots()

        x = (
            jnp.linspace(self.range[0], self.range[1], num=len(self.bins) + 1)[:-1]
            + width / 2
        )
        ax.bar(x, self.bins, width)
        plt.title(self.name)

        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()

    def plot_both(self, hist_b, save_path=None, wandb=None):
        # TO DO: Check if the relation of bins and linspace is correct
        hist_a = normalize_histogram(self.bins)
        hist_b = normalize_histogram(hist_b)

        fig, ax = plt.subplots()
        x = jnp.linspace(self.range[0], self.range[1], num=len(self.bins) + 1)[:-1]
        ax.step(x, hist_b)
        ax.step(x, hist_a)
        ax.legend(["True", "Learned"])
        plt.title(self.name)

        if save_path is not None:
            plt.savefig(save_path)
            if wandb is not None:
                # Log image(s)
                im = plt.imread(save_path)
                wandb.log({save_path: [wandb.Image(im, caption=save_path)]})
        else:
            plt.show()
        plt.close()


def normalize_histogram(hist):
    hist = jnp.array(hist)
    prob = hist / jnp.sum(hist)
    return prob


def coord2distances(x):
    x = jnp.expand_dims(x, axis=2)
    x_t = jnp.transpose(x, (0, 2, 1))
    dist = (x - x_t) ** 2
    dist = jnp.sqrt(jnp.sum(dist, axis=3))
    dist = dist.flatten()
    return dist


def earth_mover_distance(h1, h2):
    p1 = normalize_histogram(h1)
    p2 = normalize_histogram(h2)
    distance = jsp_stats.wasserstein_distance(p1, p2)
    return distance


def kl_divergence(p1, p2):
    return jnp.sum(p1 * jnp.log(p1 / p2))


def kl_divergence_sym(h1, h2):
    p1 = normalize_histogram(h1) + 1e-10
    p2 = normalize_histogram(h2) + 1e-10
    kl = kl_divergence(p1, p2)
    kl_flipped = kl_divergence(p2, p1)
    return (kl + kl_flipped) / 2.0


def js_divergence(h1, h2):
    p1 = normalize_histogram(h1) + 1e-10
    p2 = normalize_histogram(h2) + 1e-10
    M = (p1 + p2) / 2
    js = (kl_divergence(p1, M) + kl_divergence(p2, M)) / 2
    return js


def main_analyze_qm9(remove_h: bool, dataset_name="qm9", n_atoms=None):
    class DataLoaderConfig(object):
        def __init__(self):
            self.batch_size = 128
            self.remove_h = remove_h
            self.filter_n_atoms = n_atoms
            self.num_workers = 0
            self.include_charges = True
            self.dataset = (
                dataset_name  # could be qm9, qm9_first_half, or qm9_second_half
            )
            self.datadir = "qm9/temp"

    cfg = DataLoaderConfig()

    # Assuming `dataset.retrieve_dataloaders` returns similar data structures to PyTorch
    dataloaders, charge_scale = dataset.retrieve_dataloaders(cfg)

    hist_nodes = Histogram_discrete("Histogram # nodes")
    hist_atom_type = Histogram_discrete("Histogram of atom types")
    hist_dist = Histogram_cont(name="Histogram relative distances", ignore_zeros=True)

    for i, data in enumerate(dataloaders["train"]):
        print(i * cfg.batch_size)

        # Histogram num_nodes
        num_nodes = jnp.sum(data["atom_mask"], axis=1)
        num_nodes = num_nodes.tolist()
        hist_nodes.add(num_nodes)

        # Histogram edge distances
        x = data["positions"] * jnp.expand_dims(data["atom_mask"], axis=2)
        dist = coord2distances(x)
        hist_dist.add(dist.tolist())

        # Histogram of atom types
        one_hot = data["one_hot"].astype(jnp.float64)
        atom = jnp.argmax(one_hot, axis=2)
        atom = atom.flatten()
        mask = data["atom_mask"].flatten()
        masked_atoms = atom[mask].tolist()
        hist_atom_type.add(masked_atoms)

    # Plotting not supported in JAX
    hist_dist.plot()
    hist_dist.plot_both(hist_dist.bins[::-1])
    print(
        "KL divergence A %.4f" % kl_divergence_sym(hist_dist.bins, hist_dist.bins[::-1])
    )
    print("KL divergence B %.4f" % kl_divergence_sym(hist_dist.bins, hist_dist.bins))
    print(hist_dist.bins)
    hist_nodes.plot()
    print("Histogram of the number of nodes", hist_nodes.bins)
    hist_atom_type.plot()
    print(
        " Histogram of the atom types (H (optional), C, N, O, F)", hist_atom_type.bins
    )


def check_stability(positions, atom_type, dataset_info, debug=False):
    assert len(positions.shape) == 2
    assert positions.shape[1] == 3
    atom_decoder = dataset_info["atom_decoder"]
    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]

    nr_bonds = jnp.zeros(len(x), dtype=jnp.int32)

    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            p1 = jnp.array([x[i], y[i], z[i]])
            p2 = jnp.array([x[j], y[j], z[j]])
            dist = jnp.sqrt(jnp.sum((p1 - p2) ** 2))
            atom1, atom2 = atom_decoder[atom_type[i]], atom_decoder[atom_type[j]]
            pair = sorted([atom_type[i], atom_type[j]])
            if (
                dataset_info["name"] == "qm9"
                or dataset_info["name"] == "qm9_second_half"
                or dataset_info["name"] == "qm9_first_half"
            ):
                order = bond_analyze.get_bond_order(
                    atom1, atom2, dist
                )  # You need to define bond_analyze.get_bond_order
            elif dataset_info["name"] == "geom":
                order = bond_analyze.geom_predictor(
                    (atom_decoder[pair[0]], atom_decoder[pair[1]]), dist
                )  # You need to define bond_analyze.geom_predictor
            nr_bonds = jax.ops.index_add(nr_bonds, i, order)
            nr_bonds = jax.ops.index_add(nr_bonds, j, order)
    nr_stable_bonds = 0
    for atom_type_i, nr_bonds_i in zip(atom_type, nr_bonds):
        possible_bonds = bond_analyze.allowed_bonds[
            atom_decoder[atom_type_i]
        ]  # You need to define bond_analyze.allowed_bonds
        if isinstance(possible_bonds, int):
            is_stable = possible_bonds == nr_bonds_i
        else:
            is_stable = nr_bonds_i in possible_bonds
        if not is_stable and debug:
            print(
                "Invalid bonds for molecule %s with %d bonds"
                % (atom_decoder[atom_type_i], nr_bonds_i)
            )
        nr_stable_bonds += int(is_stable)

    molecule_stable = nr_stable_bonds == len(x)
    return molecule_stable, nr_stable_bonds, len(x)


def process_loader(dataloader):
    """Mask atoms, return positions and atom types"""
    out = []
    for data in dataloader:
        for i in range(data["positions"].shape[0]):
            positions = data["positions"][i].reshape(-1, 3)
            one_hot = data["one_hot"][i].reshape(-1, 5).astype(jnp.float32)
            mask = data["atom_mask"][i].flatten()
            positions, one_hot = positions[mask], one_hot[mask]
            atom_type = jnp.argmax(one_hot, axis=1)
            out.append((positions, atom_type))
    return out


def main_check_stability(remove_h: bool, batch_size=32):
    from configs import datasets_config
    import qm9.dataset as dataset

    class Config:
        def __init__(self):
            self.batch_size = batch_size
            self.num_workers = 0
            self.remove_h = remove_h
            self.filter_n_atoms = None
            self.datadir = "qm9/temp"
            self.dataset = "qm9"
            self.include_charges = True
            self.filter_molecule_size = None
            self.sequential = False

    cfg = Config()

    dataset_info = datasets_config.qm9_with_h
    dataloaders, charge_scale = dataset.retrieve_dataloaders(cfg)

    # Define use_rdkit and bond_analyze here
    if use_rdkit:
        from qm9.rdkit_functions import BasicMolecularMetrics

        metrics = BasicMolecularMetrics(dataset_info)

    atom_decoder = dataset_info["atom_decoder"]

    def test_validity_for(dataloader):
        count_mol_stable = 0
        count_atm_stable = 0
        count_mol_total = 0
        count_atm_total = 0
        for positions, atom_types in dataloader:
            is_stable, nr_stable, total = check_stability(
                positions, atom_types, dataset_info
            )

            count_atm_stable += nr_stable
            count_atm_total += total

            count_mol_stable += int(is_stable)
            count_mol_total += 1

            print(
                f"Stable molecules "
                f"{100. * count_mol_stable / count_mol_total:.2f} \t"
                f"Stable atoms: "
                f"{100. * count_atm_stable / count_atm_total:.2f} \t"
                f"Counted molecules {count_mol_total}/{len(dataloader) * batch_size}"
            )

    train_loader = process_loader(dataloaders["train"])
    test_loader = process_loader(dataloaders["test"])
    if use_rdkit:
        print("For test")
        metrics.evaluate(test_loader)
        print("For train")
        metrics.evaluate(train_loader)
    else:
        print("For train")
        test_validity_for(train_loader)
        print("For test")
        test_validity_for(test_loader)


def analyze_stability_for_molecules(molecule_list, dataset_info):
    one_hot = molecule_list["one_hot"]
    x = molecule_list["x"]
    node_mask = molecule_list["node_mask"]

    atomsxmol = (
        jnp.sum(node_mask, axis=1)
        if isinstance(node_mask, jnp.ndarray)
        else [jnp.sum(m) for m in node_mask]
    )

    n_samples = len(x)

    molecule_stable = 0
    nr_stable_bonds = 0
    n_atoms = 0

    processed_list = []

    for i in range(n_samples):
        atom_type = jnp.argmax(one_hot[i], axis=1)
        pos = x[i]

        atom_type = atom_type[: atomsxmol[i]]
        pos = pos[: atomsxmol[i]]
        processed_list.append((pos, atom_type))

    for mol in processed_list:
        pos, atom_type = mol
        validity_results = check_stability(pos, atom_type, dataset_info)

        molecule_stable += int(validity_results[0])
        nr_stable_bonds += int(validity_results[1])
        n_atoms += int(validity_results[2])

    # Validity
    fraction_mol_stable = molecule_stable / float(n_samples)
    fraction_atm_stable = nr_stable_bonds / float(n_atoms)
    validity_dict = {
        "mol_stable": fraction_mol_stable,
        "atm_stable": fraction_atm_stable,
    }

    if use_rdkit:
        metrics = BasicMolecularMetrics(dataset_info)
        rdkit_metrics = metrics.evaluate(processed_list)
        return validity_dict, rdkit_metrics
    else:
        return validity_dict, None


def analyze_node_distribution(mol_list, save_path):
    hist_nodes = Histogram_discrete("Histogram # nodes (stable molecules)")
    hist_atom_type = Histogram_discrete("Histogram of atom types")

    for molecule in mol_list:
        positions, atom_type = molecule
        hist_nodes.add([positions.shape[0]])
        hist_atom_type.add(atom_type)
    print("Histogram of #nodes")
    print(hist_nodes.bins)
    print("Histogram of # atom types")
    print(hist_atom_type.bins)
    hist_nodes.normalize()


if __name__ == "__main__":
    # Assuming main_check_stability is adapted to JAX or handled separately
    main_check_stability(remove_h=False)
