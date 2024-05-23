import torch
import numpy as np

import jax
import jax.numpy as jnp


def batch_stack(props):
    """
    Stack a list of torch.tensors so they are padded to the size of the
    largest tensor along each axis.

    Parameters
    ----------
    props : list of Pytorch Tensors
        Pytorch tensors to stack

    Returns
    -------
    props : Pytorch tensor
        Stacked pytorch tensor.

    Notes
    -----
    TODO : Review whether the behavior when elements are not tensors is safe.
    """
    if not isinstance(props[0], jnp.ndarray):
        return jnp.array(props)
    elif jnp.ndim(props[0]) == 0:
        return jnp.stack(props)
    else:
        return torch.nn.utils.rnn.pad_sequence(
            torch.tensor(np.asarray(props)), batch_first=True, padding_value=0
        )


def drop_zeros(props, to_keep):
    """
    Function to drop zeros from batches when the entire dataset is padded to the largest molecule size.

    Parameters
    ----------
    props : jax.numpy array
        Full Dataset

    to_keep : jax.numpy array
        Boolean array indicating which elements to keep.

    Returns
    -------
    props : jax.numpy array
        The dataset with only the retained information.

    Notes
    -----
    TODO : Review whether the behavior when elements are not arrays is safe.
    """
    if not isinstance(props[0], jnp.ndarray):
        return props
    elif props[0].ndim == 0:
        return props
    else:
        return props[:, to_keep, ...]


class PreprocessQM9:
    def __init__(self, load_charges=True):
        self.load_charges = load_charges

    def add_trick(self, trick):
        self.tricks.append(trick)

    def collate_fn(self, batch):
        """
        Collation function that collates datapoints into the batch format for cormorant

        Parameters
        ----------
        batch : list of datapoints
            The data to be collated.

        Returns
        -------
        batch : dict of jax.numpy arrays
            The collated data.
        """
        batch = {
            prop: batch_stack([mol[prop] for mol in batch]) for prop in batch[0].keys()
        }

        to_keep = batch["charges"].sum(0) > 0

        batch = {key: drop_zeros(prop, to_keep) for key, prop in batch.items()}

        atom_mask = batch["charges"] > 0
        batch["atom_mask"] = atom_mask

        # Obtain edges
        batch_size, n_nodes = atom_mask.shape
        edge_mask = jnp.array(
            atom_mask[:, :, jnp.newaxis] * atom_mask[:, jnp.newaxis, :]
        )

        # mask diagonal
        diag_mask = ~jnp.eye(edge_mask.shape[1], dtype=jnp.bool_)[jnp.newaxis, ...]
        edge_mask *= diag_mask

        batch["edge_mask"] = edge_mask.reshape(batch_size * n_nodes * n_nodes, 1)

        if self.load_charges:
            batch["charges"] = batch["charges"][:, :, jnp.newaxis]
        else:
            batch["charges"] = jnp.zeros(0)
        return batch
