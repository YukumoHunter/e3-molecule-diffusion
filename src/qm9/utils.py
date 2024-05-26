import jax.numpy as jnp


def compute_mean_mad(dataloaders, properties, dataset_name):
    if dataset_name == "qm9":
        return compute_mean_mad_from_dataloader(dataloaders["train"], properties)
    elif dataset_name == "qm9_second_half" or dataset_name == "qm9_second_half":
        return compute_mean_mad_from_dataloader(dataloaders["valid"], properties)
    else:
        raise Exception("Wrong dataset name")


def compute_mean_mad_from_dataloader(dataloader, properties):
    property_norms = {}
    for property_key in properties:
        values = dataloader.dataset.data[property_key]
        mean = jnp.mean(values)
        ma = jnp.abs(values - mean)
        mad = jnp.mean(ma)
        property_norms[property_key] = {}
        property_norms[property_key]["mean"] = mean
        property_norms[property_key]["mad"] = mad
    return property_norms


edges_dic = {}


def get_adj_matrix(n_nodes, batch_size, device):
    if n_nodes in edges_dic:
        edges_dic_b = edges_dic[n_nodes]
        if batch_size in edges_dic_b:
            return edges_dic_b[batch_size]
        else:
            # get edges for a single sample
            rows, cols = [], []
            for batch_idx in range(batch_size):
                for i in range(n_nodes):
                    for j in range(n_nodes):
                        rows.append(i + batch_idx * n_nodes)
                        cols.append(j + batch_idx * n_nodes)

    else:
        edges_dic[n_nodes] = {}
        return get_adj_matrix(n_nodes, batch_size, device)

    edges = [jnp.array(rows), jnp.array(cols)]
    return edges


def preprocess_input(one_hot, charges, charge_power, charge_scale, device):
    charge_tensor = (charges[..., None] / charge_scale) ** jnp.arange(
        charge_power + 1.0, dtype=jnp.float32
    )
    charge_tensor = jnp.reshape(charge_tensor, charges.shape + (1, charge_power + 1))
    atom_scalars = (one_hot[..., None] * charge_tensor).reshape(
        charges.shape[:-1] + (-1,)
    )
    return atom_scalars


def prepare_context(conditioning, minibatch, property_norms):
    batch_size, n_nodes, _ = minibatch["positions"].shape
    node_mask = jnp.expand_dims(minibatch["atom_mask"], 2)
    context_node_nf = 0
    context_list = []
    for key in conditioning:
        properties = minibatch[key]
        properties = (properties - property_norms[key]["mean"]) / property_norms[key][
            "mad"
        ]
        if properties.ndim == 1:
            # Global feature.
            assert properties.shape == (batch_size,)
            reshaped = jnp.tile(properties[:,None,None], (1,n_nodes,1))
            context_list.append(reshaped)
            context_node_nf += 1
        elif properties.ndim in [2,3]:
            # Node feature.
            assert properties.shape[:2] == (batch_size, n_nodes)

            context_key = properties

            # Inflate if necessary.
            if properties.ndim == 2:
                context_key = jnp.expand_dims(context_key, 2)

            context_list.append(context_key)
            context_node_nf += context_key.shape[2]
        else:
            raise ValueError("Invalid tensor size, more than 3 axes.")
    # Concatenate
    context = jnp.concatenate(context_list, axis=2)
    # Mask disabled nodes!
    context = context * node_mask
    assert context.shape[2] == context_node_nf
    return context
