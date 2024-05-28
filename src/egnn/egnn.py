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


def compute_radial(edge_index, x, norm_constant = 1):
    """
    Compute x_i - x_j and ||x_i - x_j||^2.
    """
    senders, receivers = edge_index
    x_i, x_j = x[senders], x[receivers]
    coord_diff = x_i - x_j
    radial = jnp.sum((coord_diff) ** 2, axis=1, keepdims=True)
    norm = jnp.sqrt(radial + 1e-8)
    coord_diff = coord_diff / (norm + norm_constant)
    return radial, coord_diff


def custom_xavier_uniform_init(gain=0.001):
    """
    Low variance initialization used in positional MLPs
    """

    def init(key, shape, dtype=jnp.float32):
        std = gain * jnp.sqrt(2.0 / shape[0])
        return jax.random.uniform(key, shape, dtype, -std, std)

    return init

class AttMLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        out = nn.Sequential(
            nn.Dense(1),
            nn.sigmoid(),
            )(x)
        return out
    
def build_fn(hidden_dim, act_fn, attention):
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

        if attention:
            att_val = nn.Sequential([
                nn.Dense(1),
                nn.sigmoid()])(out)
            out = out * att_val

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
    norm_constant = 1
    attention: bool = True

    @nn.compact
    def __call__(self, edge_index, h, x, edge_attr, node_mask, edge_mask):  #EquivariantBlock
        # get primitives
        message_fn, agg_update_fn, pos_agg_update_fn = build_fn(self.hidden_dim, self.act_fn, self.attention)
        # compute the distance between connected nodes
        dist = compute_radial(edge_index, x, norm_constant=self.norm_constant)
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
    out_dim: int = None
    num_layers: int
    act_fn: callable = jax.nn.silu

    def setup(self):
        if self.out_dim is None:
            self.out_dim = self.hidden_dim

        self.initial_dense = nn.Dense(self.hidden_dim)
        self.egnn_layers = [EGNN_layer(self.hidden_dim, self.act_fn) for _ in range(self.num_layers)]
        self.output_dense = nn.Dense(self.out_dim)

    def __call__(self, edge_index, h, x, node_mask=None, edge_mask=None):
        distances = compute_radial(edge_index, x)  # Assuming compute_radial is defined elsewhere
        h = self.initial_dense(h)

        for layer in self.egnn_layers:
            h, x = layer(edge_index, h, x, edge_attr=distances, node_mask=node_mask, edge_mask=edge_mask)

        h = self.output_dense(h)
        if node_mask is not None:
            h = h * node_mask
        return h, x