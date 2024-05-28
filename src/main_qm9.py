# Rdkit import should be first, do not move it
try:
    from rdkit import Chem
except ModuleNotFoundError:
    pass
import copy
import utils
import argparse
import wandb
import numpy as np
from configs.datasets_config import get_dataset_info
from os.path import join
from qm9 import dataset
from qm9.models import get_optim, get_model
from equivariant_diffusion import en_diffusion
from equivariant_diffusion.utils import assert_correctly_masked
from equivariant_diffusion import utils as flow_utils
import time
import pickle
from qm9.utils import prepare_context, compute_mean_mad
# from train_test import train_epoch, test, analyze_and_save
from train_test import create_train_step_and_state, create_test_step, test



import jax
import jax.numpy as jnp
from jax import device_put
from jax import random
from jax import jit
from jax.scipy.special import logsumexp
from jax.nn import softplus
from jax.nn.initializers import uniform, variance_scaling, kaiming_uniform
import flax.linen as nn

from jax import lax


parser = argparse.ArgumentParser(description="E3Diffusion")
parser.add_argument("--exp_name", type=str, default="debug_10")
parser.add_argument(
    "--model",
    type=str,
    default="egnn_dynamics",
    help="our_dynamics | schnet | simple_dynamics | "
    "kernel_dynamics | egnn_dynamics |gnn_dynamics",
)
parser.add_argument(
    "--probabilistic_model", type=str, default="diffusion", help="diffusion"
)

# Training complexity is O(1) (unaffected), but sampling complexity is O(steps).
parser.add_argument("--diffusion_steps", type=int, default=500)
parser.add_argument(
    "--diffusion_noise_schedule",
    type=str,
    default="polynomial_2",
    help="learned, cosine",
)
parser.add_argument(
    "--diffusion_noise_precision",
    type=float,
    default=1e-5,
)
parser.add_argument("--diffusion_loss_type", type=str, default="l2", help="vlb, l2")

parser.add_argument("--n_epochs", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=128) #64
parser.add_argument("--lr", type=float, default=2e-4) #1e-4
parser.add_argument("--brute_force", type=eval, default=False, help="True | False")
parser.add_argument("--actnorm", type=eval, default=True, help="True | False")
parser.add_argument(
    "--break_train_epoch", type=eval, default=False, help="True | False"
)
parser.add_argument("--dp", type=eval, default=True, help="True | False")
parser.add_argument("--condition_time", type=eval, default=True, help="True | False")
parser.add_argument("--clip_grad", type=eval, default=True, help="True | False")
parser.add_argument("--trace", type=str, default="hutch", help="hutch | exact")
# EGNN args -->
parser.add_argument("--n_layers", type=int, default=6, help="number of layers") #9
parser.add_argument("--inv_sublayers", type=int, default=1, help="number of layers")
parser.add_argument("--nf", type=int, default=128, help="number of features") #256
parser.add_argument("--tanh", type=eval, default=True, help="use tanh in the coord_mlp")
parser.add_argument(
    "--attention", type=eval, default=True, help="use attention in the EGNN"
)
parser.add_argument(
    "--norm_constant", type=float, default=1, help="diff/(|diff| + norm_constant)"
)
parser.add_argument(
    "--sin_embedding",
    type=eval,
    default=False,
    help="whether using or not the sin embedding",
)
# <-- EGNN args
parser.add_argument("--ode_regularization", type=float, default=1e-3)
parser.add_argument(
    "--dataset",
    type=str,
    default="qm9",
    help="qm9 | qm9_second_half (train only on the last 50K samples of the training dataset)",
)
parser.add_argument("--datadir", type=str, default="qm9/temp", help="qm9 directory")
parser.add_argument(
    "--filter_n_atoms",
    type=int,
    default=None,
    help="When set to an integer value, QM9 will only contain molecules of that amount of atoms",
)
parser.add_argument(
    "--dequantization",
    type=str,
    default="argmax_variational",
    help="uniform | variational | argmax_variational | deterministic",
)
parser.add_argument("--n_report_steps", type=int, default=1)
parser.add_argument("--wandb_usr", type=str)
parser.add_argument("--no_wandb", action="store_true", help="Disable wandb")
parser.add_argument(
    "--online",
    type=bool,
    default=True,
    help="True = wandb online -- False = wandb offline",
)
parser.add_argument(
    "--no-cuda", action="store_true", default=False, help="enables CUDA training"
)
parser.add_argument("--save_model", type=eval, default=True, help="save model")
parser.add_argument("--generate_epochs", type=int, default=1, help="save model")
parser.add_argument(
    "--num_workers", type=int, default=0, help="Number of worker for the dataloader"
)
parser.add_argument("--test_epochs", type=int, default=10) #20
parser.add_argument(
    "--data_augmentation", type=eval, default=False, help="use attention in the EGNN"
)
parser.add_argument(
    "--conditioning",
    nargs="+",
    default=[],
    help="arguments : homo | lumo | alpha | gap | mu | Cv",
)
parser.add_argument("--resume", type=str, default=None, help="")
parser.add_argument("--start_epoch", type=int, default=0, help="")
parser.add_argument(
    "--ema_decay",
    type=float,
    default=0.999,
    help="Amount of EMA decay, 0 means off. A reasonable value" " is 0.999.",
) #0.9999
parser.add_argument("--augment_noise", type=float, default=0)
parser.add_argument(
    "--n_stability_samples",
    type=int,
    default=500,
    help="Number of samples to compute the stability",
)
parser.add_argument(
    "--normalize_factors",
    type=eval,
    default=[1, 4, 1],
    help="normalize factors for [x, categorical, integer]",
)
parser.add_argument("--remove_h", action="store_true")
parser.add_argument(
    "--include_charges", type=eval, default=True, help="include atom charge or not"
)
parser.add_argument(
    "--visualize_every_batch",
    type=int,
    default=1e8,
    help="Can be used to visualize multiple times per epoch",
)
parser.add_argument(
    "--normalization_factor",
    type=float,
    default=1,
    help="Normalize the sum aggregation of EGNN",
)
parser.add_argument(
    "--aggregation_method", type=str, default="sum", help='"sum" or "mean"'
)
args = parser.parse_args()

dataset_info = get_dataset_info(args.dataset, args.remove_h)

atom_encoder = dataset_info["atom_encoder"]
atom_decoder = dataset_info["atom_decoder"]

# args, unparsed_args = parser.parse_known_args()
args.wandb_usr = utils.get_wandb_username(args.wandb_usr)

dtype = jnp.float32

if args.resume is not None:
    exp_name = args.exp_name + "_resume"
    start_epoch = args.start_epoch
    resume = args.resume
    wandb_usr = args.wandb_usr
    normalization_factor = args.normalization_factor
    aggregation_method = args.aggregation_method

    with open(join(args.resume, "args.pickle"), "rb") as f:
        args = pickle.load(f)

    args.resume = resume
    args.break_train_epoch = False

    args.exp_name = exp_name
    args.start_epoch = start_epoch
    args.wandb_usr = wandb_usr

    # Careful with this -->
    if not hasattr(args, "normalization_factor"):
        args.normalization_factor = normalization_factor
    if not hasattr(args, "aggregation_method"):
        args.aggregation_method = aggregation_method

    print(args)

utils.create_folders(args)
# print(args)


# Wandb config
if args.no_wandb:
    mode = "disabled"
else:
    mode = "online" if args.online else "offline"
kwargs = {
    "entity": args.wandb_usr,
    "name": args.exp_name,
    "project": "e3_diffusion",
    "config": args,
    "settings": wandb.Settings(_disable_stats=True),
    "reinit": True,
    "mode": mode,
}
wandb.init(**kwargs)
wandb.save("*.txt")

# Retrieve QM9 dataloaders
dataloaders, charge_scale = dataset.retrieve_dataloaders(args)

data_dummy = next(iter(dataloaders["train"]))


if len(args.conditioning) > 0:
    print(f"Conditioning on {args.conditioning}")
    property_norms = compute_mean_mad(dataloaders, args.conditioning, args.dataset)
    context_dummy = prepare_context(args.conditioning, data_dummy, property_norms)
    context_node_nf = context_dummy.shape[2]
else:
    context_node_nf = 0
    property_norms = None

args.context_node_nf = context_node_nf

SEED = 42

key = random.key(SEED)
key, subkey = random.split(key)

# Create EGNN flow
model, nodes_dist, prop_dist = get_model(
    subkey, args, dataset_info, dataloaders["train"]
) #if args.conditionining == [] then prop_dist = None
if prop_dist is not None:
    prop_dist.set_normalizer(property_norms)
optim = get_optim(args, model) #not initialized yet
# print(model)

# gradnorm_queue = utils.Queue()
# gradnorm_queue.add(3000)  # Add large value that will be flushed.

# ? it is not used
def check_mask_correct(variables, node_mask):
    for variable in variables:
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)


def main():
    if args.resume is not None:
        flow_state_dict = jnp.load(join(args.resume, "flow.npy"))
        optim_state_dict = jnp.load(join(args.resume, "optim.npy"))
        model.load_state_dict(flow_state_dict)
        optim.load_state_dict(optim_state_dict)

    model_dp = model

    # Initialize model copy for exponential moving average of params.
    if args.ema_decay > 0:
        model_ema = copy.deepcopy(model)
        ema = flow_utils.EMA(args.ema_decay) #old * self.beta + (1 - self.beta) * new

        model_ema_dp = model_ema
    else:
        ema = None
        model_ema = model
        model_ema_dp = model_dp

    training_step_jitted, state = create_train_step_and_state(key, model, optim, dataloaders["train"], nodes_dist, args)    
    test_step = create_test_step(args, nodes_dist)
    best_nll_val = 1e8
    best_nll_test = 1e8
    times_forward_backwards = []
    for epoch in range(args.start_epoch, args.n_epochs):
        nll_epoch = []
        start_epoch = time.time()
        n_iterations = len(dataloaders["train"])
        for i, batch in enumerate(dataloaders["train"]):
            state, loss, nll, reg_term, time_fb_batch = training_step_jitted(state, batch)
            times_forward_backwards.append(time_fb_batch)

            if i % args.n_report_steps == 0:
                print(
                    f"\rEpoch: {epoch}, iter: {i}/{n_iterations}, "
                    f"Loss {loss.item():.2f}, NLL: {nll.item():.2f}, "
                    f"RegTerm: {reg_term.item():.1f}, "
                    f"GradNorm: {state.opt_state[3].items[0]:.1f}" #TODO: Is this correct??
                )
            nll_epoch.append(nll.item())
            # if (
            #     (epoch % args.test_epochs == 0)
            #     and (i % args.visualize_every_batch == 0)
            #     and not (epoch == 0 and i == 0)
            # ):
            #     start = time.time()
            #     if len(args.conditioning) > 0:
            #         save_and_sample_conditional(
            #             args, model_ema, prop_dist, dataset_info, epoch=epoch
            #         )
            #     save_and_sample_chain(
            #         model_ema, args, dataset_info, prop_dist, epoch=epoch, batch_id=str(i)
            #     )
            #     sample_different_sizes_and_save(
            #         model_ema, nodes_dist, args, dataset_info, prop_dist, epoch=epoch
            #     )
            #     print(f"Sampling took {time.time() - start:.2f} seconds")

            #     vis.visualize(
            #         f"outputs/{args.exp_name}/epoch_{epoch}_{i}",
            #         dataset_info=dataset_info,
            #         wandb=wandb,
            #     )
            #     vis.visualize_chain(
            #         f"outputs/{args.exp_name}/epoch_{epoch}_{i}/chain/",
            #         dataset_info,
            #         wandb=wandb,
            #     )
            #     if len(args.conditioning) > 0:
            #         vis.visualize_chain(
            #             "outputs/%s/epoch_%d/conditional/" % (args.exp_name, epoch),
            #             dataset_info,
            #             wandb=wandb,
            #             mode="conditional",
            #         )
            # wandb.log({"Batch NLL": nll.item()}, commit=True)
            # if args.break_train_epoch:
            #     break
            # wandb.log({"Train Epoch NLL": np.mean(nll_epoch)}, commit=False)
        
        print(f"Epoch took {time.time() - start_epoch:.1f} seconds.")
        print(f"    forward + backward passes took-> mean:{np.mean([times_forward_backwards])}")
        if epoch % args.test_epochs == 0:
            # if isinstance(model, en_diffusion.EnVariationalDiffusion):
            #     wandb.log(model.log_info(), commit=True)

            # if not args.break_train_epoch:
            #     analyze_and_save(
            #         args=args,
            #         epoch=epoch,
            #         model_sample=model_ema,
            #         nodes_dist=nodes_dist,
            #         dataset_info=dataset_info,
            #         prop_dist=prop_dist,
            #         n_samples=args.n_stability_samples,
            #     )
            nll_val = test(
                state, 
                test_step, 
                test_loader = dataloaders["valid"],
                epoch = epoch, 
                args = args, 
                partition = "Validation"
                ) 
            nll_test = test(
                state, 
                test_step, 
                test_loader = dataloaders["test"],
                epoch = epoch, 
                args = args, 
                partition = "Validation"
                ) 
            
            def save_model(params, filepath):
                with open(filepath, 'wb') as f:
                    f.write(flax.serialization.to_bytes(params))

            if nll_val < best_nll_val:
                best_nll_val = nll_val
                best_nll_test = nll_test
                if save_model:
                    args.current_epoch = epoch + 1
                    os.makedirs(f"outputs/{exp_name}", exist_ok=True)
                    save_model(optimizer_state, f""outputs/%s/optim.npy" % args.exp_name")
                    save_model(model_params, f"outputs/{exp_name}/generative_model.npy")
                    if ema_decay > 0:
                        save_model(model_params, f"outputs/{exp_name}/generative_model_ema.npy")
                    with open(f"outputs/{exp_name}/args.pickle", "wb") as f:
                        pickle.dump(args, f)

                if save_model:
                    save_model(optimizer_state, f"outputs/{exp_name}/optim_{epoch}.npy")
                    save_model(model_params, f"outputs/{exp_name}/generative_model_{epoch}.npy")
                    if ema_decay > 0:
                        save_model(model_params, f"outputs/{exp_name}/generative_model_ema_{epoch}.npy")
                    with open(f"outputs/{exp_name}/args_{epoch}.pickle", "wb") as f:
                        pickle.dump(args, f)

            print("Val loss: %.4f \t Test loss:  %.4f" % (nll_val, nll_test))
            print("Best val loss: %.4f \t Best test loss:  %.4f" % (best_nll_val, best_nll_test))

            # Log to wandb
            wandb.log({"Val loss": nll_val}, commit=True)
            wandb.log({"Test loss": nll_test}, commit=True)
            wandb.log({"Best cross-validated test loss": best_nll_test}, commit=True)


            # if nll_val < best_nll_val:
            #     best_nll_val = nll_val
            #     best_nll_test = nll_test
            #     if args.save_model:
            #         args.current_epoch = epoch + 1
            #         utils.save_model(optim, "outputs/%s/optim.npy" % args.exp_name)
            #         utils.save_model(
            #             model, "outputs/%s/generative_model.npy" % args.exp_name
            #         )
            #         if args.ema_decay > 0:
            #             utils.save_model(
            #                 model_ema,
            #                 "outputs/%s/generative_model_ema.npy" % args.exp_name,
            #             )
            #         with open("outputs/%s/args.pickle" % args.exp_name, "wb") as f:
            #             pickle.dump(args, f)

            #     if args.save_model:
            #         utils.save_model(
            #             optim, "outputs/%s/optim_%d.npy" % (args.exp_name, epoch)
            #         )
            #         utils.save_model(
            #             model,
            #             "outputs/%s/generative_model_%d.npy" % (args.exp_name, epoch),
            #         )
            #         if args.ema_decay > 0:
            #             utils.save_model(
            #                 model_ema,
            #                 "outputs/%s/generative_model_ema_%d.npy"
            #                 % (args.exp_name, epoch),
            #             )
            #         with open(
            #             "outputs/%s/args_%d.pickle" % (args.exp_name, epoch), "wb"
            #         ) as f:
            #             pickle.dump(args, f)
            # print("Val loss: %.4f \t Test loss:  %.4f" % (nll_val, nll_test))
            # print(
            #     "Best val loss: %.4f \t Best test loss:  %.4f"
            #     % (best_nll_val, best_nll_test)
            # )
            # wandb.log({"Val loss ": nll_val}, commit=True)
            # wandb.log({"Test loss ": nll_test}, commit=True)
            # wandb.log({"Best cross-validated test loss ": best_nll_test}, commit=True)


if __name__ == "__main__":
    main()
