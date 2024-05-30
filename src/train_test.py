import wandb
from equivariant_diffusion.utils import (
    assert_mean_zero_with_mask,
    remove_mean_with_mask,
    assert_correctly_masked,
    sample_center_gravity_zero_gaussian_with_mask,
)
import numpy as np
import qm9.visualizer as vis
from qm9.analyze import analyze_stability_for_molecules
from qm9.sampling import sample_chain, sample, sample_sweep_conditional
import utils
import qm9.utils as qm9utils
from qm9 import losses
import time
import tqdm
import torch
import flax
from flax.training import train_state

from typing import Any

import jax
import jax.numpy as jnp
from jax import random
import optax

# def create_train_step_and_state(key, model, optim, dataloader, nodes_dist, args):
#     data = next(iter(dataloader))
#     x = data["positions"]
#     node_mask = jnp.expand_dims(data["atom_mask"], 2)

#     print("node mask: ", data["atom_mask"].shape)

#     edge_mask = data["edge_mask"]
#     one_hot = data["one_hot"]
#     charges = data["charges"] if args.include_charges else jnp.zeros(0)
#     h = {"categorical": one_hot, "integer": charges}
#     context = None
#     bs, n_nodes, n_dims = x.shape
#     edge_mask = jnp.reshape(edge_mask, (bs, n_nodes * n_nodes))
#     params = model.init(key, x, h, node_mask, edge_mask, context)

#     state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optim)

#     @jax.jit
#     def train_step(state, batch):
#         def loss_fn(params, nodes_dist, x, h, node_mask, edge_mask, context):
#             bs, n_nodes, _ = x.shape
#             edge_mask = jnp.reshape(edge_mask, (bs, n_nodes * n_nodes))
#             nll = state.appy_fn(params, x, h, node_mask, edge_mask, context)

#             N = jnp.sum(node_mask.squeeze(axis=2), axis=1).astype(jnp.int64)
#             log_pN = nodes_dist.log_prob(N)
#             nll = nll - log_pN
#             nll = nll.mean(0)
#             reg_term = jnp.array([0.0])
#             mean_abs_z = 0.0
#             loss = nll + args.ode_regularization * reg_term
#             return loss, (nll, reg_term)

#         x = batch["positions"]

#         node_mask = jnp.expand_dims(batch["atom_mask"], 2)
#         edge_mask = batch["edge_mask"]
#         one_hot = batch["one_hot"]
#         charges = batch["charges"] if args.include_charges else jnp.zeros(0)

#         x = remove_mean_with_mask(x, node_mask)
#         h = {"categorical": one_hot, "integer": charges}
#         context = None
#         value_grad = jax.value_and_grad(loss_fn, has_aux=True)
#         t0 = time.time()
#         losses, grads = value_grad(
#             state.params, nodes_dist, x, h, node_mask, edge_mask, context
#         )
#         time_forward_backwards = time.time() - t0
#         loss, (nll, reg_term) = losses

#         state = state.apply_gradients(grads=grads)

#         return state, loss, nll, reg_term, time_forward_backwards

#     return train_step, state


def create_train_step_and_state(key, model, optim, dataloader, nodes_dist, args):
    data = next(iter(dataloader))
    x = data["positions"]
    node_mask = jnp.expand_dims(data["atom_mask"], 2)

    # print("node mask: ", data["atom_mask"].shape)

    edge_mask = data["edge_mask"]
    one_hot = data["one_hot"]
    charges = data["charges"] if args.include_charges else jnp.zeros(0)
    h = {"categorical": one_hot, "integer": charges}
    context = None
    bs, n_nodes, n_dims = x.shape
    edge_mask = jnp.reshape(edge_mask, (bs, n_nodes * n_nodes))
    variables = model.init(key, x, h, node_mask, edge_mask, context, True)
    params = variables["params"]
    mutable_variables = variables["mutable_variables"]
    opt_state = optim.init(params)

    class TrainState(train_state.TrainState):
        mutable_variables: flax.core.FrozenDict[str, Any]

    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optim,
        mutable_variables=mutable_variables,
    )

    # @jax.jit
    def train_step(rng_key, state, opt_state, batch):
        def loss_fn(params, rng_key, nodes_dist, x, h, node_mask, edge_mask, context):
            bs, n_nodes, _ = x.shape
            edge_mask = jnp.reshape(edge_mask, (bs, n_nodes * n_nodes))
            nll, updated_state = state.apply_fn(
                {"params": params, "mutable_variables": state.mutable_variables},
                x,
                h,
                node_mask,
                edge_mask,
                context,
                True,
                rngs={"params": rng_key},
                mutable=["mutable_variables"],
            )

            N = jnp.sum(node_mask.squeeze(axis=2), axis=1).astype(jnp.int32)

            log_pN = nodes_dist.log_prob(N)

            nll = nll - log_pN
            nll = nll.mean(0)

            # print("log_pN: ", log_pN)

            reg_term = jnp.array([0.0])
            # mean_abs_z = 0.0
            loss = nll + args.ode_regularization * reg_term
            loss = loss.item()
            return loss, (nll, reg_term, updated_state)

        x = batch["positions"]

        node_mask = jnp.expand_dims(batch["atom_mask"], 2)
        edge_mask = batch["edge_mask"]
        one_hot = batch["one_hot"]
        charges = batch["charges"] if args.include_charges else jnp.zeros(0)

        x = remove_mean_with_mask(x, node_mask)
        h = {"categorical": one_hot, "integer": charges}
        context = None
        value_grad = jax.value_and_grad(loss_fn, has_aux=True, allow_int=True)

        t0 = time.time()

        (loss, (nll, reg_term, new_params)), grads = value_grad(
            state.params, key, nodes_dist, x, h, node_mask, edge_mask, context
        )
        # updates, opt_state = state.tx.update(grads, opt_state, state.mutable_variables)
        # new_params = optax.apply_updates(new_params, updates)

        time_forward_backwards = time.time() - t0

        state = state.apply_gradients(grads=grads)
        # state.replace(mutable_variables=grads["mutable_variables"], params=new_params)

        return state, loss, nll, reg_term, time_forward_backwards, opt_state

    return train_step, state, opt_state


def create_test_step(args, nodes_dist):
    @jax.jit
    def test_step(state, batch):
        def loss_fn(params, x, h, node_mask, edge_mask, context):
            bs, n_nodes, n_dims = x.shape
            edge_mask = jnp.reshape(edge_mask, (bs, n_nodes * n_nodes))
            nll = state.apply_fn(params, x, h, node_mask, edge_mask, context)

            N = jnp.sum(node_mask.squeeze(axis=2), axis=1).astype(jnp.int32)
            log_pN = nodes_dist.log_prob(N)
            nll = nll - log_pN
            nll = nll.mean(0)
            return nll

        x = batch["positions"]
        batch_size = x.shape[0]
        node_mask = jnp.expand_dims(batch["atom_mask"], 2)
        edge_mask = batch["edge_mask"]
        one_hot = batch["one_hot"]
        charges = batch["charges"] if args.include_charges else jnp.zeros(0)

        x = remove_mean_with_mask(x, node_mask)
        h = {"categorical": one_hot, "integer": charges}
        context = None

        nll = loss_fn(state.params, x, h, node_mask, edge_mask, context)
        cum_nll = nll.item() * batch_size
        n_samples = batch_size
        return cum_nll, n_samples

    return test_step


def test(state, test_step, test_loader, epoch, args, partition="Test"):
    cum_nll_epoch = 0
    n_samples = 0
    n_iterations = len(test_loader)

    for i, data in enumerate(test_loader):
        cum_nll_batch, n_samples_batch = test_step(state, data)

        cum_nll_epoch += cum_nll_batch
        n_samples += n_samples_batch
        if i % args.n_report_steps == 0:
            print(
                f"\r {partition} NLL \t epoch: {epoch}, iter: {i}/{n_iterations}, "
                f"NLL: {cum_nll_epoch/n_samples:.2f}"
            )

    return cum_nll_epoch / n_samples


#############################
# ORIGINAL
# I removed device
# def train_epoch(
#     args,
#     loader,
#     epoch,
#     model,
#     model_dp,
#     model_ema,
#     ema,
#     dtype,
#     property_norms, #None
#     optim,
#     nodes_dist,
#     # gradnorm_queue,
#     dataset_info,
#     prop_dist, #None
#     rng
# ):
#     # model.train()
#     nll_epoch = []
#     n_iterations = len(loader)
#     for i, data in enumerate(loader):
#         x = data["positions"]
#         node_mask = jnp.expand_dims(data["atom_mask"], 2)
#         edge_mask = data["edge_mask"]
#         one_hot = data["one_hot"]
#         charges = data["charges"] if args.include_charges else jnp.zeros(0)

#         x = remove_mean_with_mask(x, node_mask)

#         if args.augment_noise > 0:
#             rng, key_sample_gaussian = random.split(rng, 2)
#             # Add noise eps ~ N(0, augment_noise) around points.
#             eps = sample_center_gravity_zero_gaussian_with_mask(key_sample_gaussian, x.shape, x, node_mask)
#             x = x + eps * args.augment_noise

#         x = remove_mean_with_mask(x, node_mask)
#         if args.data_augmentation: #defaul false
#             x = utils.random_rotation(x)

#         check_mask_correct([x, one_hot, charges], node_mask)
#         assert_mean_zero_with_mask(x, node_mask)

#         h = {"categorical": one_hot, "integer": charges}

#         if len(args.conditioning) > 0: #default None
#             context = qm9utils.prepare_context(args.conditioning, data, property_norms)
#             assert_correctly_masked(context, node_mask)
#         else:
#             context = None

#         optim.zero_grad()

#         # transform batch through flow
#         nll, reg_term, mean_abs_z = losses.compute_loss_and_nll(
#             args, model_dp, nodes_dist, x, h, node_mask, edge_mask, context
#         )
#         # standard nll from forward KL
#         loss = nll + args.ode_regularization * reg_term
#         loss.backward()

#         # if args.clip_grad:
#         #     grad_norm = utils.gradient_clipping(model, gradnorm_queue)
#         # else:
#         #     grad_norm = 0.0

#         optim.step()

#         # Update EMA if enabled.
#         # if args.ema_decay > 0:
#         #     ema.update_model_average(model_ema, model)

#         if i % args.n_report_steps == 0:
#             print(
#                 f"\rEpoch: {epoch}, iter: {i}/{n_iterations}, "
#                 f"Loss {loss.item():.2f}, NLL: {nll.item():.2f}, "
#                 f"RegTerm: {reg_term.item():.1f}, "
#                 f"GradNorm: {grad_norm:.1f}"
#             )
#         nll_epoch.append(nll.item())
#         if (
#             (epoch % args.test_epochs == 0)
#             and (i % args.visualize_every_batch == 0)
#             and not (epoch == 0 and i == 0)
#         ):
#             start = time.time()
#             if len(args.conditioning) > 0:
#                 save_and_sample_conditional(
#                     args, model_ema, prop_dist, dataset_info, epoch=epoch
#                 )
#             save_and_sample_chain(
#                 model_ema, args, dataset_info, prop_dist, epoch=epoch, batch_id=str(i)
#             )
#             sample_different_sizes_and_save(
#                 model_ema, nodes_dist, args, dataset_info, prop_dist, epoch=epoch
#             )
#             print(f"Sampling took {time.time() - start:.2f} seconds")

#             vis.visualize(
#                 f"outputs/{args.exp_name}/epoch_{epoch}_{i}",
#                 dataset_info=dataset_info,
#                 wandb=wandb,
#             )
#             vis.visualize_chain(
#                 f"outputs/{args.exp_name}/epoch_{epoch}_{i}/chain/",
#                 dataset_info,
#                 wandb=wandb,
#             )
#             if len(args.conditioning) > 0:
#                 vis.visualize_chain(
#                     "outputs/%s/epoch_%d/conditional/" % (args.exp_name, epoch),
#                     dataset_info,
#                     wandb=wandb,
#                     mode="conditional",
#                 )
#         wandb.log({"Batch NLL": nll.item()}, commit=True)
#         if args.break_train_epoch:
#             break
#     wandb.log({"Train Epoch NLL": np.mean(nll_epoch)}, commit=False)


def check_mask_correct(variables, node_mask):
    for i, variable in enumerate(variables):
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)


# I removed device
# def test(
#     args, loader, epoch, eval_model, dtype, property_norms, nodes_dist, partition="Test"
# ):
#     eval_model.eval()
#     with torch.no_grad():
#         nll_epoch = 0
#         n_samples = 0

#         n_iterations = len(loader)

#         for i, data in enumerate(loader):
#             x = data["positions"]
#             batch_size = x.size(0)
#             node_mask = data["atom_mask"].unsqueeze(2)
#             edge_mask = data["edge_mask"]
#             one_hot = data["one_hot"]
#             charges = data["charges"] if args.include_charges else torch.zeros(0)

#             if args.augment_noise > 0:
#                 # Add noise eps ~ N(0, augment_noise) around points.
#                 eps = sample_center_gravity_zero_gaussian_with_mask(
#                     x.size(), x, node_mask
#                 )
#                 x = x + eps * args.augment_noise

#             x = remove_mean_with_mask(x, node_mask)
#             check_mask_correct([x, one_hot, charges], node_mask)
#             assert_mean_zero_with_mask(x, node_mask)

#             h = {"categorical": one_hot, "integer": charges}

#             if len(args.conditioning) > 0:
#                 context = qm9utils.prepare_context(
#                     args.conditioning, data, property_norms
#                 )
#                 assert_correctly_masked(context, node_mask)
#             else:
#                 context = None

#             # transform batch through flow
#             nll, _, _ = losses.compute_loss_and_nll(
#                 args, eval_model, nodes_dist, x, h, node_mask, edge_mask, context
#             )
#             # standard nll from forward KL

#             nll_epoch += nll.item() * batch_size
#             n_samples += batch_size
#             if i % args.n_report_steps == 0:
#                 print(
#                     f"\r {partition} NLL \t epoch: {epoch}, iter: {i}/{n_iterations}, "
#                     f"NLL: {nll_epoch/n_samples:.2f}"
#                 )

#     return nll_epoch / n_samples


# Removed device
def save_and_sample_chain(
    model, args, dataset_info, prop_dist, epoch=0, id_from=0, batch_id=""
):
    one_hot, charges, x = sample_chain(
        args=args, flow=model, n_tries=1, dataset_info=dataset_info, prop_dist=prop_dist
    )

    vis.save_xyz_file(
        f"outputs/{args.exp_name}/epoch_{epoch}_{batch_id}/chain/",
        one_hot,
        charges,
        x,
        dataset_info,
        id_from,
        name="chain",
    )

    return one_hot, charges, x


# Removed device
def sample_different_sizes_and_save(
    model,
    nodes_dist,
    args,
    dataset_info,
    prop_dist,
    n_samples=5,
    epoch=0,
    batch_size=100,
    batch_id="",
):
    batch_size = min(batch_size, n_samples)
    for counter in range(int(n_samples / batch_size)):
        nodesxsample = nodes_dist.sample(batch_size)
        one_hot, charges, x, node_mask = sample(
            args,
            model,
            prop_dist=prop_dist,
            nodesxsample=nodesxsample,
            dataset_info=dataset_info,
        )
        print(f"Generated molecule: Positions {x[:-1, :, :]}")
        vis.save_xyz_file(
            f"outputs/{args.exp_name}/epoch_{epoch}_{batch_id}/",
            one_hot,
            charges,
            x,
            dataset_info,
            batch_size * counter,
            name="molecule",
        )


# I removed device
def analyze_and_save(
    epoch,
    model_sample,
    nodes_dist,
    args,
    dataset_info,
    prop_dist,
    n_samples=1000,
    batch_size=100,
):
    print(f"Analyzing molecule stability at epoch {epoch}...")
    batch_size = min(batch_size, n_samples)
    assert n_samples % batch_size == 0
    molecules = {"one_hot": [], "x": [], "node_mask": []}
    for i in range(int(n_samples / batch_size)):
        nodesxsample = nodes_dist.sample(batch_size)
        one_hot, charges, x, node_mask = sample(
            args, model_sample, dataset_info, prop_dist, nodesxsample=nodesxsample
        )

        molecules["one_hot"].append(one_hot)
        molecules["x"].append(x)
        molecules["node_mask"].append(node_mask)

    molecules = {key: torch.cat(molecules[key], dim=0) for key in molecules}
    validity_dict, rdkit_tuple = analyze_stability_for_molecules(
        molecules, dataset_info
    )

    wandb.log(validity_dict)
    if rdkit_tuple is not None:
        wandb.log(
            {
                "Validity": rdkit_tuple[0][0],
                "Uniqueness": rdkit_tuple[0][1],
                "Novelty": rdkit_tuple[0][2],
            }
        )
    return validity_dict


def save_and_sample_conditional(
    args, model, prop_dist, dataset_info, epoch=0, id_from=0
):
    one_hot, charges, x, node_mask = sample_sweep_conditional(
        args, model, dataset_info, prop_dist
    )

    vis.save_xyz_file(
        "outputs/%s/epoch_%d/conditional/" % (args.exp_name, epoch),
        one_hot,
        charges,
        x,
        dataset_info,
        id_from,
        name="conditional",
        node_mask=node_mask,
    )

    return one_hot, charges, x
