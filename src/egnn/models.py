import jax
import jax.numpy as jnp
import flax.linen as nn

# from egnn.egnn import EGNN, GNN
from egnn.egnn import EGNN
from equivariant_diffusion.utils import remove_mean, remove_mean_with_mask


class EGNN_dynamics_QM9(nn.Module):
    in_node_nf: int
    context_node_nf: int
    n_dims: int
    hidden_nf: int = 64
    act_fn: callable = jax.nn.silu
    n_layers: int = 4
    attention: bool = False
    condition_time: bool = True
    tanh: bool = False
    mode: str = "egnn_dynamics"
    norm_constant: int = 0
    inv_sublayers: int = 2
    sin_embedding: bool = False
    normalization_factor: int = 100
    aggregation_method: str = "sum"

    # 27
    def setup(self):
        is_initialized = self.has_variable("mutable_variables", "edges_dict")
        # print("is_initialized:", is_initialized)
        self.edges_dict = self.variable("mutable_variables", "edges_dict", lambda: {})
        # print("is_initialized after:", is_initialized)

        if self.mode == "egnn_dynamics":
            self.egnn = EGNN(
                in_node_nf=self.in_node_nf + self.context_node_nf,
                in_edge_nf=1,
                hidden_nf=self.hidden_nf,
                act_fn=self.act_fn,
                n_layers=self.n_layers,
                attention=self.attention,
                tanh=self.tanh,
                norm_constant=self.norm_constant,
                inv_sublayers=self.inv_sublayers,
                sin_embedding=self.sin_embedding,
                normalization_factor=self.normalization_factor,
                aggregation_method=self.aggregation_method,
            )
        elif self.mode == "gnn_dynamics":
            raise NotImplementedError

            # self.gnn = GNN(
            #     in_node_nf=self.in_node_nf + self.context_node_nf + 3,
            #     in_edge_nf=0,
            #     hidden_nf=self.hidden_nf,
            #     out_node_nf=3 + self.in_node_nf,
            #     act_fn=self.act_fn,
            #     n_layers=self.n_layers,
            #     attention=self.attention,
            #     normalization_factor=self.normalization_factor,
            #     aggregation_method=self.aggregation_method,
            # )

    def forward(self, t, xh, node_mask, edge_mask, context=None):
        raise NotImplementedError

    def wrap_forward(self, node_mask, edge_mask, context):
        def fwd(time, state):
            return self._forward(time, state, node_mask, edge_mask, context)

        return fwd

    def unwrap_forward(self):
        return self._forward

    def __call__(self, node_mask, edge_mask, context=None):
        # print(
        #     "EGNN dynamics:",
        #     type(node_mask),
        #     type(edge_mask),
        #     type(context),
        # )
        return self.wrap_forward(node_mask, edge_mask, context)

    def _forward(self, t, xh, node_mask, edge_mask, context):
        bs, n_nodes, dims = xh.shape
        h_dims = dims - self.n_dims
        edges = self.get_adj_matrix(n_nodes, bs)

        # print("forward dynamics:", type(bs), type(n_nodes))

        node_mask = node_mask.reshape(bs * n_nodes, 1)
        edge_mask = edge_mask.reshape(bs * n_nodes * n_nodes, 1)
        xh = xh.reshape(bs * n_nodes, -1) * node_mask
        x = xh[:, 0 : self.n_dims]
        if h_dims == 0:
            h = jnp.ones(bs * n_nodes, 1)
        else:
            h = xh[:, self.n_dims :]

        if self.condition_time:
            prod = jnp.prod(jnp.array(t.shape))
            if prod == 1:
                # t is the same for all elements in batch.
                h_time = jnp.empty_like(h[:, 0:1]).fill_(t.item())
            else:
                # t is different over the batch dimension.
                h_time = t.reshape(bs, 1)

                # print(f"h_time initial: {h_time.shape}")

                h_time = h_time.repeat(n_nodes, axis=1)

                # print(f"h_time repeated: {h_time.shape}")

                h_time = h_time.reshape(bs * n_nodes, 1)

                # print(f"h_time final: {h_time.shape}")
            h = jnp.concatenate([h, h_time], axis=1)

        if context is not None:
            # We're conditioning, awesome!
            context = context.reshape(bs * n_nodes, self.context_node_nf)
            h = jnp.concatenate([h, context], axis=1)

        if self.mode == "egnn_dynamics":
            # h_final, x_final = self.egnn(
            #     edges, h, x, node_mask=node_mask, edge_mask=edge_mask
            # )
            h_final, x_final = self.egnn(
                h, x, edges, node_mask=node_mask, edge_mask=edge_mask
            )
            vel = (
                x_final - x
            ) * node_mask  # This masking operation is redundant but just in case
        elif self.mode == "gnn_dynamics":
            xh = jnp.concatenate([x, h], axis=1)
            output = self.gnn(xh, edges, node_mask=node_mask)
            vel = output[:, 0:3] * node_mask
            h_final = output[:, 3:]

        else:
            raise Exception("Wrong mode %s" % self.mode)

        if context is not None:
            # Slice off context size:
            h_final = h_final[:, : -self.context_node_nf]

        if self.condition_time:
            # Slice off last dimension which represented time.
            h_final = h_final[:, :-1]

        vel = vel.reshape(bs, n_nodes, -1)

        if jnp.isnan(vel).any():
            print("Warning: detected nan, resetting EGNN output to zero.")
            vel = jnp.zeros_like(vel)

        if node_mask is None:
            vel = remove_mean(vel)
        else:
            vel = remove_mean_with_mask(vel, node_mask.reshape(bs, n_nodes, 1))

        if h_dims == 0:
            return vel
        else:
            h_final = h_final.reshape(bs, n_nodes, -1)
            return jnp.concatenate([vel, h_final], axis=2)

    def get_adj_matrix(self, n_nodes, batch_size):
        edges_dict = self.variables["mutable_variables"]["edges_dict"]

        if n_nodes in edges_dict:
            edges_dic_b = edges_dict[n_nodes]
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
                edges = [
                    jnp.array(rows, dtype=jnp.int32),
                    jnp.array(cols, dtype=jnp.int32),
                ]
                edges_dic_b[batch_size] = edges
                return edges
        else:
            edges_dict[n_nodes] = {}
            return self.get_adj_matrix(n_nodes, batch_size)
