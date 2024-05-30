import jax
import jax.numpy as jnp
from flax import linen as nn
import math
from typing import Any

class GCL(nn.Module):
    input_nf: int
    output_nf: int
    hidden_nf: int
    normalization_factor: float
    aggregation_method: str
    edges_in_d: int = 0
    nodes_att_dim: int = 0
    act_fn: Any = nn.silu
    attention: bool = True

    def setup(self):
        self.edge_mlp = nn.Sequential([
            nn.Dense(self.hidden_nf),
            self.act_fn,
            nn.Dense(self.hidden_nf),
            self.act_fn
        ])

        self.node_mlp = nn.Sequential([
            nn.Dense(self.hidden_nf + self.input_nf + self.nodes_att_dim),
            self.act_fn,
            nn.Dense(self.output_nf)
        ])

        if self.attention:
            self.att_mlp = nn.Sequential([
                nn.Dense(1),
                nn.sigmoid
            ])

    def edge_model(self, source, target, edge_attr, edge_mask):
        if edge_attr is None:
            out = jnp.concatenate([source, target], axis=1)
        else:
            out = jnp.concatenate([source, target, edge_attr], axis=1)
        mij = self.edge_mlp(out)

        if self.attention:
            att_val = self.att_mlp(mij)
            out = mij * att_val
        else:
            out = mij

        if edge_mask is not None:
            out = out * edge_mask
        return out, mij

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.shape[0],
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method)
        if node_attr is not None:
            agg = jnp.concatenate([x, agg, node_attr], axis=1)
        else:
            agg = jnp.concatenate([x, agg], axis=1)
        out = x + self.node_mlp(agg)
        return out, agg

    def __call__(self, h, edge_index, edge_attr=None, node_attr=None, node_mask=None, edge_mask=None):
        row, col = edge_index
        edge_feat, mij = self.edge_model(h[row], h[col], edge_attr, edge_mask)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        if node_mask is not None:
            h = h * node_mask
        return h, mij


class EquivariantUpdate(nn.Module):
    hidden_nf: int
    normalization_factor: float
    aggregation_method: str
    edges_in_d: int = 1
    act_fn: Any = nn.silu
    tanh: bool = False
    coords_range: float = 10.0

    def setup(self):
        self.coord_mlp = nn.Sequential([
            nn.Dense(self.hidden_nf),
            self.act_fn,
            nn.Dense(self.hidden_nf),
            self.act_fn,
            nn.Dense(1, use_bias=False, kernel_init=nn.initializers.variance_scaling(scale=0.001**2, mode = "fan_avg", distribution="uniform"))
        ])

    def coord_model(self, h, coord, edge_index, coord_diff, edge_attr, edge_mask):
        row, col = edge_index
        input_tensor = jnp.concatenate([h[row], h[col], edge_attr], axis=1)
        if self.tanh:
            trans = coord_diff * jnp.tanh(self.coord_mlp(input_tensor)) * self.coords_range
        else:
            trans = coord_diff * self.coord_mlp(input_tensor)
        if edge_mask is not None:
            trans = trans * edge_mask
        agg = unsorted_segment_sum(trans, row, num_segments=coord.shape[0],
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method)
        coord = coord + agg
        return coord

    def __call__(self, h, coord, edge_index, coord_diff, edge_attr=None, node_mask=None, edge_mask=None):
        coord = self.coord_model(h, coord, edge_index, coord_diff, edge_attr, edge_mask)
        if node_mask is not None:
            coord = coord * node_mask
        return coord


class EquivariantBlock(nn.Module):
    hidden_nf: int
    edge_feat_nf: int = 2
    act_fn: Any = nn.silu
    n_layers: int = 2
    attention: bool = True
    norm_diff: bool = True
    tanh: bool = False
    coords_range: float = 15
    norm_constant: float = 1
    sin_embedding: bool = False
    normalization_factor: float = 100
    aggregation_method: str = 'sum'

    def setup(self):
        self.gcls = [GCL(
            self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=self.edge_feat_nf,
            act_fn=self.act_fn, attention=self.attention,
            normalization_factor=self.normalization_factor,
            aggregation_method=self.aggregation_method) for _ in range(self.n_layers)]
        
        self.gcl_equiv = EquivariantUpdate(
            self.hidden_nf, edges_in_d=self.edge_feat_nf, act_fn=nn.silu, tanh=self.tanh,
            coords_range=self.coords_range,
            normalization_factor=self.normalization_factor,
            aggregation_method=self.aggregation_method)

    def __call__(self, h, x, edge_index, node_mask=None, edge_mask=None, edge_attr=None):
        distances, coord_diff = coord2diff(x, edge_index, self.norm_constant)
        if self.sin_embedding:
            distances = SinusoidsEmbeddingNew()(distances)
        edge_attr = jnp.concatenate([distances, edge_attr], axis=1)
        for gcl in self.gcls:
            h, _ = gcl(h, edge_index, edge_attr=edge_attr, node_mask=node_mask, edge_mask=edge_mask)
        x = self.gcl_equiv(h, x, edge_index, coord_diff, edge_attr, node_mask, edge_mask)

        if node_mask is not None:
            h = h * node_mask
        return h, x


class EGNN(nn.Module):
    in_node_nf: int
    in_edge_nf: int
    hidden_nf: int
    act_fn: Any = nn.silu
    n_layers: int = 3
    attention: bool = False
    norm_diff: bool = True
    out_node_nf: int = None
    tanh: bool = False
    coords_range: float = 15
    norm_constant: float = 1
    inv_sublayers: int = 2
    sin_embedding: bool = False
    normalization_factor: float = 100
    aggregation_method: str = 'sum'

    def setup(self):
        out_node_nf = self.out_node_nf or self.in_node_nf

        self.coords_range_layer = float(self.coords_range / self.n_layers)

        # if self.sin_embedding:
        #     self.sin_embedding = SinusoidsEmbeddingNew()
        #     edge_feat_nf = self.sin_embedding.dim * 2
        # else:
        #     self.sin_embedding = None
        edge_feat_nf = 2

        self.embedding = nn.Dense(self.hidden_nf)
        self.embedding_out = nn.Dense(out_node_nf)
        
        self.e_blocks = [EquivariantBlock(
            self.hidden_nf, edge_feat_nf=edge_feat_nf,
            act_fn=self.act_fn, n_layers=self.inv_sublayers,
            attention=self.attention, norm_diff=self.norm_diff, tanh=self.tanh,
            coords_range=self.coords_range, norm_constant=self.norm_constant,
            sin_embedding=self.sin_embedding,
            normalization_factor=self.normalization_factor,
            aggregation_method=self.aggregation_method) for _ in range(self.n_layers)]

    def __call__(self, h, x, edge_index, node_mask=None, edge_mask=None):
        distances, _ = coord2diff(x, edge_index)
        if self.sin_embedding:
            distances = self.sin_embedding(distances)
        h = self.embedding(h)
        for e_block in self.e_blocks:
            h, x = e_block(h, x, edge_index, node_mask=node_mask, edge_mask=edge_mask, edge_attr=distances)

        h = self.embedding_out(h)
        if node_mask is not None:
            h = h * node_mask
        return h, x

class SinusoidsEmbeddingNew(nn.Module):
    max_res: float = 15.0
    min_res: float = 15.0 / 2000.0
    div_factor: int = 4

    def setup(self):
        self.n_frequencies = int(math.log(self.max_res / self.min_res, self.div_factor)) + 1
        self.frequencies = 2 * jnp.pi * self.div_factor ** jnp.arange(self.n_frequencies) / self.max_res
        self.dim = len(self.frequencies) * 2

    def __call__(self, x):
        x = jnp.sqrt(x + 1e-8)
        emb = x[:, None] * self.frequencies
        emb = jnp.concatenate((jnp.sin(emb), jnp.cos(emb)), axis=-1)
        return emb


def coord2diff(x, edge_index, norm_constant=1):
    row, col = edge_index
    coord_diff = x[row] - x[col]
    radial = jnp.sum((coord_diff) ** 2, axis=1, keepdims=True)
    norm = jnp.sqrt(radial + 1e-8)
    coord_diff = coord_diff / (norm + norm_constant)
    return radial, coord_diff


def unsorted_segment_sum(data, segment_ids, num_segments, normalization_factor, aggregation_method: str):
    result_shape = (num_segments, data.shape[1])
    result = jnp.zeros(result_shape, dtype=data.dtype)
    # segment_ids = jnp.expand_dims(segment_ids, -1).repeat(data.shape[1], axis=-1)
    result = jax.ops.segment_sum(data, segment_ids, num_segments)
    if aggregation_method == 'sum':
        result = result / normalization_factor
    elif aggregation_method == 'mean':
        norm = jax.ops.segment_sum(jnp.ones_like(data), segment_ids, num_segments)
        norm = jnp.where(norm == 0, 1, norm)
        result = result / norm
    return result
