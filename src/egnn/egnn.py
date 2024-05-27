import jax
import jax.numpy as jnp
import flax.linen as nn


def segment_mean(data, segment_ids, num_segments):
    """
    Computes the mean within segments of an array.
    """
    # Sum the data within each segment
    segment_sums = jax.ops.segment_sum(data, segment_ids, num_segments)
    # Compute the size of each segment
    segment_sizes = jax.ops.segment_sum(jnp.ones_like(data), segment_ids, num_segments)
    segment_means = segment_sums / segment_sizes
    return segment_means


def compute_radial(edge_index, x):
    """
    Compute x_i - x_j and ||x_i - x_j||^2.
    """
    senders, receivers = edge_index
    x_i, x_j = x[senders], x[receivers]
    distance = jnp.sum((x_i - x_j) ** 2, axis=1, keepdims=True)
    return distance


def custom_xavier_uniform_init(gain=0.001):
    """
    Low variance initialization used in positional MLPs
    """

    def init(key, shape, dtype=jnp.float32):
        std = gain * jnp.sqrt(2.0 / shape[0])
        return jax.random.uniform(key, shape, dtype, -std, std)

    return init


def build_fn(hidden_dim, act_fn):
    """
    EGNN primitives as functions
        1. message function (eq. 3)
        2. message aggregation + node update (eq. 5,6)
        3. message aggregation + positional update (eq. 4)
    """
    # 27
    def message_fn(edge_index, h, dist, edge_attr, edge_mask): #edge_model
        """
        Message: m_ij = phi_e(h_i^l, h_j^l, ||x_i^l - x_j^l||^2, a_ij)
        """
        phi_e = nn.Sequential(
            [nn.Dense(hidden_dim), act_fn, nn.Dense(hidden_dim), act_fn]  #is like edge_mlp
        )

        senders, receivers = edge_index
        h_i, h_j = h[senders], h[receivers]
        out = jnp.concatenate([h_i, h_j, dist, edge_attr], axis=1)

        out = phi_e(out)

        if edge_mask is not None:
            out = out * edge_mask

        return out

    def agg_update_fn(edge_index, h_i, m_ij):
        """
        Aggregation: m_i = sum_{j!=i} m_ij

        Node update: h_i^{l+1} = phi_h(h_i^l, m_i)
        """
        phi_h = nn.Sequential([nn.Dense(hidden_dim), act_fn, nn.Dense(hidden_dim)]) #node_mlp

        senders, _ = edge_index
        m_i = jax.ops.segment_sum(m_ij, senders, num_segments=h_i.shape[0])
        out = jnp.concatenate([h_i, m_i], axis=1)

        return h_i + phi_h(out)

    def pos_agg_update_fn(edge_index, x, m_ij, node_mask): #EquivariantUpdate
        """
        Positional update: x_i^{l+1} = x_i^l + mean_{j!=i} (x_i^l - x_j^l) phi_x(m_ij)
        """
        phi_x = nn.Sequential( #coord_mlp
            [
                nn.Dense(hidden_dim),
                act_fn,
                nn.Dense(1, kernel_init=custom_xavier_uniform_init(gain=0.001)),
            ]
        )

        senders, receivers = edge_index
        x_i, x_j = x[senders], x[receivers]
        x_ij = (x_i - x_j) * phi_x(m_ij)

        coord = x + segment_mean(x_ij, senders, num_segments=x.shape[0])

        if node_mask is not None:
            coord = coord * node_mask

        return coord

    return message_fn, agg_update_fn, pos_agg_update_fn


class EGNN_layer(nn.Module):
    hidden_dim: int
    act_fn: callable


    @nn.compact
    def __call__(self, edge_index, h, x, edge_attr, node_mask, edge_mask):  #EquivariantBlock
        # get primitives
        message_fn, agg_update_fn, pos_agg_update_fn = build_fn(self.hidden_dim, self.act_fn)
        # compute the distance between connected nodes
        dist = compute_radial(edge_index, x)
        # message -> aggregation -> node update, position update
        #GCL
        m_ij = message_fn(edge_index, h, dist, edge_attr, edge_mask)
        h = agg_update_fn(edge_index, h, m_ij)
        if node_mask is not None:
            h = h * node_mask
        
        #EquivariantUpdate
        x = pos_agg_update_fn(edge_index, x, m_ij, node_mask)

        if node_mask is not None:
            h = h * node_mask
        return h, x


class EGNN(nn.Module):
    hidden_dim: int
    out_dim = None
    num_layers: int 
    act_fn: callable = jax.nn.silu

    if out_dim is None:
        out_dim = hidden_dim

    @nn.compact
    def __call__(self, edge_index, h, x, node_mask = None, edge_mask = None):
        
        distances = compute_radial(edge_index, x) #27
        h = nn.Dense(self.hidden_dim)(h)

        for _ in range(self.num_layers):
            h, x = EGNN_layer(self.hidden_dim, self.act_fn)(edge_index, h, x, edge_attr = distances, node_mask = node_mask, edge_mask = edge_mask)

        h = nn.Dense(self.out_dim)(h)
        if node_mask is not None:
            h = h * node_mask
        return h, x
