import jax
import jax.numpy as jnp
import optax
from optax._src import combine, transform
import jax.random as random

import numpy as np
from egnn.models import EGNN_dynamics_QM9
from src.utils import AdamW_with_amsgrad
from equivariant_diffusion.en_diffusion import EnVariationalDiffusion


def get_model(args, dataset_info, dataloader_train):
    histogram = dataset_info["n_nodes"]
    in_node_nf = len(dataset_info["atom_decoder"]) + int(args.include_charges)
    nodes_dist = DistributionNodes(histogram)

    prop_dist = None
    if len(args.conditioning) > 0:
        prop_dist = DistributionProperty(dataloader_train, args.conditioning)

    if args.condition_time:
        dynamics_in_node_nf = in_node_nf + 1
    else:
        print("Warning: dynamics model is _not_ conditioned on time.")
        dynamics_in_node_nf = in_node_nf

    net_dynamics = EGNN_dynamics_QM9(
        in_node_nf=dynamics_in_node_nf,
        context_node_nf=args.context_node_nf,
        n_dims=3,
        hidden_nf=args.nf,
        act_fn=jax.nn.silu,
        n_layers=args.n_layers,
        attention=args.attention,
        tanh=args.tanh,
        mode=args.model,
        norm_constant=args.norm_constant,
        inv_sublayers=args.inv_sublayers,
        sin_embedding=args.sin_embedding,
        normalization_factor=args.normalization_factor,
        aggregation_method=args.aggregation_method,
    )

    if args.probabilistic_model == "diffusion":
        vdm = EnVariationalDiffusion(
            dynamics=net_dynamics,
            in_node_nf=in_node_nf,
            n_dims=3,
            timesteps=args.diffusion_steps,
            noise_schedule=args.diffusion_noise_schedule,
            noise_precision=args.diffusion_noise_precision,
            loss_type=args.diffusion_loss_type,
            norm_values=args.normalize_factors,
            include_charges=args.include_charges,
        )

        return vdm, nodes_dist, prop_dist

    else:
        raise ValueError(args.probabilistic_model)


def get_optim(args):
    optimizer = AdamW_with_amsgrad(learning_rate=args.lr, weight_decay=1e-12, gradnorm_queue = 50)
    return optimizer


class DistributionNodes:
    def __init__(self, histogram):
        self.n_nodes = []
        prob = []
        self.keys = {}
        for i, nodes in enumerate(histogram):
            self.n_nodes.append(nodes)
            self.keys[nodes] = i
            prob.append(histogram[nodes])
        self.n_nodes = jnp.array(self.n_nodes)

        prob = jnp.array(prob)
        prob = prob / jnp.sum(prob)

        self.prob = prob.astype(jnp.float32)

        entropy = jnp.sum(self.prob * jnp.log(self.prob + 1e-30))
        print("Entropy of n_nodes: H[N]", entropy.item())

        # key, subkey = random.split(rng_key)

        # self.m = random.categorical(key=subkey, logits=prob)

    def sample(self, rng, n_samples=1):

        idx = random.choice(rng, len(self.n_nodes), shape = (n_samples,), p = self.prob)
        # idx = random.categorical(key=rng, logits=self.prob)
        return self.n_nodes[idx]

    def log_prob(self, batch_n_nodes):
        assert batch_n_nodes.ndim == 1

        idcs = [self.keys[i.item()] for i in batch_n_nodes]
        idcs = jnp.array(idcs).to(batch_n_nodes)

        log_p = jnp.log(self.prob + 1e-30)

        log_p = log_p.to(batch_n_nodes)

        log_probs = log_p[idcs]

        return log_probs


class DistributionProperty:
    def __init__(self, dataloader, properties, num_bins=1000, normalizer=None):
        self.num_bins = num_bins
        self.distributions = {}
        self.properties = properties
        for prop in properties:
            self.distributions[prop] = {}
            self._create_prob_dist(
                dataloader.dataset.data["num_atoms"],
                dataloader.dataset.data[prop],
                self.distributions[prop],
            )

        self.normalizer = normalizer

    def set_normalizer(self, normalizer):
        self.normalizer = normalizer

    def _create_prob_dist(self, nodes_arr, values, distribution):
        min_nodes, max_nodes = jnp.min(nodes_arr), jnp.max(nodes_arr)
        for n_nodes in range(int(min_nodes), int(max_nodes) + 1):
            idxs = nodes_arr == n_nodes
            values_filtered = values[idxs]
            if len(values_filtered) > 0:
                probs, params = self._create_prob_given_nodes(values_filtered)
                distribution[n_nodes] = {"probs": probs, "params": params}

    def _create_prob_given_nodes(self, rng_key, values):
        n_bins = self.num_bins  # min(self.num_bins, len(values))
        prop_min, prop_max = jnp.min(values), jnp.max(values)
        prop_range = prop_max - prop_min + 1e-12
        histogram = jnp.zeros(n_bins)
        for val in values:
            i = int((val - prop_min) / prop_range * n_bins)
            # Because of numerical precision, one sample can fall in bin int(n_bins) instead of int(n_bins-1)
            # We move it to bin int(n_bind-1 if tat happens)
            if i == n_bins:
                i = n_bins - 1
            # histogram[i] += 1
            histogram = histogram.at[i].add(1)

        probs = histogram / jnp.sum(histogram)
        # self.probs = probs

        # key, subkey = random.split(rng_key)

        # probs = random.categorical(key=subkey, logits=probs)
        params = [prop_min, prop_max]
        return probs, params

    def normalize_tensor(self, tensor, prop):
        assert self.normalizer is not None
        mean = self.normalizer[prop]["mean"]
        mad = self.normalizer[prop]["mad"]
        return (tensor - mean) / mad

    def sample(self, rng, n_nodes=19):
        vals = []
        for prop in self.properties:
            dist = self.distributions[prop][n_nodes]
            rng, subkey1, subkey2 = random.split(rng, 3)
            idx = random.categorical(key=subkey1, logits=jnp.log(dist["probs"]))
            val = self._idx2value(subkey2, idx, dist["params"], len(dist["probs"]))
            val = self.normalize_tensor(val, prop)
            vals.append(val)
        vals = jnp.concatenate(vals)
        return vals

    def sample_batch(self, rng, nodesxsample):
        vals = []
        for n_nodes in nodesxsample:
            rng, subkey = random.split(rng)
            vals.append(jnp.expand_dims(self.sample(subkey,int(n_nodes)), 0))
        vals = jnp.concatenate(vals, dim=0)
        return vals

    def _idx2value(self, key, idx, params, n_bins):
        prop_range = params[1] - params[0]
        left = float(idx) / n_bins * prop_range + params[0]
        right = float(idx + 1) / n_bins * prop_range + params[0]
        val = random.uniform(key, (1,), minval=left, maxval=right)
        return val


if __name__ == "__main__":
    dist_nodes = DistributionNodes()
    print(dist_nodes.n_nodes)
    print(dist_nodes.prob)
    for i in range(10):
        print(dist_nodes.sample(random.key(0)))
