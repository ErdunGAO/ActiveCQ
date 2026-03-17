
import torch
from torch import cuda

import ray
import click

from pathlib import Path

import sys
import os
os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))

from src.application import workflows

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True)


@click.group(chain=True)
@click.pass_context
def cli(context):
    context.obj = {"n_gpu": cuda.device_count()}

def validate_value(ctx, param, value):
    if value == "None":
        return None
    try:
        return float(value)
    except ValueError:
        raise click.BadParameter("condition_value must be 'None' or a valid float.")




@cli.command("active-learning")
@click.option(
    "--job-dir",
    default="experiments",
    type=str,
    required=True,
    help="location for writing checkpoints and results",
)
@click.option(
    "--num-trials",
    default=1,
    type=int,
    help="number of trials, default=1",
)
@click.option(
    "--acq-size",
    default=10,
    type=int,
    help="number of data points to acquire at each step, default=10",
)
@click.option(
    "--warm-start-size",
    default=100,
    type=int,
    help="number of data points to acquire at start, default=100",
)
@click.option(
    "--max-acquisitions",
    default=38,
    type=int,
    help="number of acquisition steps, default=100",
)
@click.option(
    "--acquisition-function",
    default="random",
    type=str,
    help="acquistion function, default=random, current[random, coresets, var_rank, var_reduction_rank, ACQE]",
)
@click.option(
    "--adaptive-strategy",
    default="IG",
    type=str,
    help="adaptive strategy, default=IG, we have [IG, VR]",
)
@click.option(
    "--cde-estimator",
    default="CME",
    type=str,
    help="cde estimator, default=cme, we have [cme, mdn, lscde]",
)
@click.option(
    "--batch-aware",
    default="B",
    type=str,
    help="batch learner, default=batch, we have B for batch, G for greedy",
)
@click.option(
    "--exp_mode",
    default="run",
    type=str,
    help="experiment mode, default=debug",
)
@click.option(
    "--verbose",
    default=False,
    type=bool,
    help="verbosity default=False")
@click.option(
    "--seed",
    default=2024,
    type=int,
    help="random number generator seed, default=2024",
)
@click.pass_context
def active_learning(
    context,
    job_dir,
    num_trials,
    acq_size,
    warm_start_size,
    max_acquisitions,
    acquisition_function,
    adaptive_strategy,
    cde_estimator,
    batch_aware,
    exp_mode,
    verbose,
    seed,
):
    # ray.init(
    #     num_gpus=context.obj["n_gpu"],
    #     dashboard_host="127.0.0.1",
    #     ignore_reinit_error=True,
    #     object_store_memory=8000000000,
    #     local_mode=True,
    #     include_dashboard=False
    # )
    if exp_mode == "debug":
        job_dir = "debug" # double checking here
    job_dir = (
        Path(job_dir)
        / "active_learning"
    )
    context.obj.update(
        {
            "job_dir": str(job_dir),
            "num_trials": num_trials,
            "acq_size": acq_size,
            "warm_start_size": warm_start_size,
            "max_acquisitions": max_acquisitions,
            "acquisition_function": acquisition_function,
            "adaptive_strategy": adaptive_strategy,
            "cde_estimator": cde_estimator,
            "batch_aware": batch_aware,
            "exp_mode": exp_mode,
            "verbose": verbose,
            "seed": seed,
        }
    )


@cli.command("evaluate")
@click.option(
    "--experiment-dir",
    type=str,
    required=True,
    help="location for reading checkpoints",
)
@click.option(
    "--output-dir",
    type=str,
    required=False,
    default=None,
    help="location for writing results",
)
@click.pass_context
def evaluate(
    context, experiment_dir, output_dir,
):
    output_dir = experiment_dir if output_dir is None else output_dir
    context.obj.update(
        {"experiment_dir": experiment_dir, "output_dir": output_dir}
    )


@cli.command("ihdp")
@click.pass_context
@click.option(
    "--root", type=str,
    default='data/ihdp',
    required=True, help="location of dataset",
)
@click.option(
    "--task_type",
    default="cate",
    type=str,
    help="task type, [cate, ate, scate, att], default=cate",
)
@click.option(
    "--treatment_type",
    default="binary",
    type=str,
    help="task type, [binary, continuous], default=cate",
)
@click.option(
    "--condition_value",
    default="None",
    callback=validate_value,
    help="Value of condition variable, default=None. Can be None or a float.",
)
@click.option(
    "--treatment_value",
    default="None",
    callback=validate_value,
    help="value of treatment variable, default=None, \
        if we set None, we will randomly generate the value",
)
@click.option(
    "--treatment_interest",
    default=False,
    type=bool,
    help="has interested treatment variable, default=False, \
        if we set True, we are all treatments, \
        if we set False, we will randomly generate one specific value",
)
@click.option(
    "--condition_name",
    default="bw",
    type=str,
    help="which covariate to choose, default=sex",
)
@click.option(
    "--condition_type",
    default="discrete",
    type=str,
    help="type of the condition variable, [binary,discrete, continuous], default=dicrete",
)
@click.option(
    "--regression_func",
    default="easy",
    type=str,
    help="type of the regression function, [easy, hard], default=easy",
)
@click.option(
    "--treatment_assignment",
    default="weak",
    type=str,
    help="type of the treatment assignment, [weak, strong], default=weak",
)
def ihdp(
    context, root, task_type, treatment_type, condition_value, treatment_value, treatment_interest, condition_name, condition_type, regression_func, treatment_assignment,
):
    job_dir = Path(context.obj.get("job_dir"))
    dataset_name = "ihdp"
    job_dir = context.obj.get("job_dir")
    if job_dir is not None:
        experiment_dir = (
            Path(job_dir)
        )
    else:
        experiment_dir = None
    context.obj.update(
        {
            "dataset_name": dataset_name,
            "experiment_dir": str(experiment_dir),
            "treatment_type": treatment_type,
            "condition_value": condition_value,
            "treatment_value": treatment_value,
            "task_type": task_type,
            "treatment_interest": treatment_interest,
            "condition_name": condition_name,
            "condition_type": condition_type,
            "regression_func": regression_func,
            "treatment_assignment": treatment_assignment,
            "ds_train": {
                "root": root,
                "treatment_type": treatment_type,
                "task_type": task_type,
                "treatment_type": treatment_type,
                "condition_value": condition_value,
                "treatment_value": treatment_value,
                "treatment_interest": treatment_interest,
                "condition_name": condition_name,
                "condition_type": condition_type,
                "dataset_type": "train",
                "regression_func": regression_func,
                "treatment_assignment": treatment_assignment,
                "seed": context.obj.get("seed"),
            },
            "ds_valid": {
                "root": root,
                "treatment_type": treatment_type,
                "task_type": task_type,
                "treatment_type": treatment_type,
                "condition_value": condition_value,
                "treatment_value": treatment_value,
                "treatment_interest": treatment_interest,
                "condition_name": condition_name,
                "condition_type": condition_type,
                "dataset_type": "valid",
                "regression_func": regression_func,
                "treatment_assignment": treatment_assignment,
                "seed": context.obj.get("seed"),
            },
            "ds_test": {
                "root": root,
                "treatment_type": treatment_type,
                "task_type": task_type,
                "treatment_type": treatment_type,
                "condition_value": condition_value,
                "treatment_value": treatment_value,
                "treatment_interest": treatment_interest,
                "condition_name": condition_name,
                "condition_type": condition_type,
                "dataset_type": "test",
                "regression_func": regression_func,
                "treatment_assignment": treatment_assignment,
                "seed": context.obj.get("seed"),
            },
        }
    )


@cli.command("simulation")
@click.pass_context
@click.option(
    "--num-examples",
    default=600,
    type=int,
    help="number of training examples, defaul=1000",
)
@click.option(
    "--task_type",
    default="cate",
    type=str,
    help="task type, [cate, ate, scate, att], default=cate",
)
@click.option(
    "--treatment_type",
    default="binary",
    type=str,
    help="task type, [binary, continuous], default=cate",
)
@click.option(
    "--dim-adjustment",
    default=4,
    type=int,
    help="number of adjustment variables, default=4",
)
@click.option(
    "--condition_value",
    default="None",
    callback=validate_value,
    help="Value of condition variable, default=None. Can be None or a float.",
)
@click.option(
    "--treatment_value",
    default="None",
    callback=validate_value,
    help="value of treatment variable, default=None, \
        if we set None, we will randomly generate the value",
)
@click.option(
    "--treatment_interest",
    default=False,
    type=bool,
    help="has interested treatment variable, default=False, \
        if we set True, we are all treatments, \
        if we set False, we will randomly generate one specific value",
)
@click.option(
    "--condition_type",
    default="continuous",
    type=str,
    help="type of the condition variable, [discrete, continuous], default=continuous",
)
@click.option(
    "--regression_func",
    default="middle",
    type=str,
    help="type of the regression function, [easy, middle, hard], default=middle",
)
@click.option(
    "--treatment_assignment",
    default="weak",
    type=str,
    help="type of the treatment assignment, [weak, strong], default=weak",
)
@click.option(
    "--condition_dist",
    default="hard",
    type=str,
    help="type of the treatment assignment, [easy, hard], default=hard",
)
def simulation(
    context, num_examples, task_type, treatment_type, dim_adjustment, condition_value, treatment_value, treatment_interest, condition_type, regression_func, treatment_assignment, condition_dist,
):
    dataset_name = "simulation"
    job_dir = context.obj.get("job_dir")
    if job_dir is not None:
        experiment_dir = (
            Path(job_dir)
        )
    else:
        experiment_dir = None

    context.obj.update(
        {
            "dataset_name": dataset_name,
            "experiment_dir": str(experiment_dir),
            "condition_type": condition_type,
            "treatment_type": treatment_type,
            "condition_value": condition_value,
            "treatment_value": treatment_value,
            "task_type": task_type,
            "treatment_interest": treatment_interest,
            "regression_func": regression_func,
            "treatment_assignment": treatment_assignment,
            "condition_dist": condition_dist,
            "ds_train": {
                "num_examples": num_examples,
                "task_type": task_type,
                "treatment_type": treatment_type,
                "dim_adjustment": dim_adjustment,
                "condition_type": condition_type,
                "condition_value": condition_value,
                "treatment_value": treatment_value,
                "treatment_interest": treatment_interest,
                "dataset_type": "train",
                "regression_func": regression_func,
                "treatment_assignment": treatment_assignment,
                "condition_dist": condition_dist,
                "seed": context.obj.get("seed", 0),
            },
            "ds_valid": {
                "num_examples": 200,
                "task_type": task_type,
                "treatment_type": treatment_type,
                "dim_adjustment": dim_adjustment,
                "condition_type": condition_type,
                "condition_value": condition_value,
                "treatment_value": treatment_value,
                "treatment_interest": treatment_interest,
                "dataset_type": "valid",
                "regression_func": regression_func,
                "treatment_assignment": treatment_assignment,
                "condition_dist": condition_dist,
                "seed": context.obj.get("seed", 0) + 1,
            },
            "ds_test": {
                "num_examples": 200,
                "task_type": task_type,
                "treatment_type": treatment_type,
                "dim_adjustment": dim_adjustment,
                "condition_type": condition_type,
                "condition_value": condition_value,
                "treatment_value": treatment_value,
                "treatment_interest": treatment_interest,
                "dataset_type": "test",
                "regression_func": regression_func,
                "treatment_assignment": treatment_assignment,
                "condition_dist": condition_dist,
                "seed": context.obj.get("seed", 0) + 2,
            },
        }
    )


@cli.command("imp")
@click.pass_context
@click.option(
    "--learning-rate",
    default=0.05,
    type=float,
    help="learning rate for gradient descent, default=1e-3",
)
@click.option(
    "--gp_epochs",
    type=int,
    default=500,
    help="number of training epochs, default=500"
)
@click.option(
    "--cme_epochs",
    type=int,
    default=50,
    help="number of training epochs, default=500"
)
@click.option(
    "--kernel_type",
    type=str,
    default="rbf",
    help="kernel type, default=rbf, we have [rbf, matern, rq]",
)
@click.option(
    "--learn_cme",
    type=bool,
    default=False,
    help="Whether learn the CME."
)
@click.option(
    "--device",
    type=str,
    default="gpu",
    help="device to run model on, default=gpu",
)
def imp(
    context,
    learning_rate,
    learn_cme,
    gp_epochs,
    cme_epochs,
    kernel_type,
    device,
):
    context.obj.update(
        {
            "learning_rate": learning_rate,
            "gp_epochs": gp_epochs,
            "cme_epochs": cme_epochs,
            "learn_cme": learn_cme,
            "kernel_type": kernel_type,
            "device": device,
        }
    )

    results = []
    for trial in range(context.obj.get("num_trials")):
        result = workflows.active_learning.active_learner(
            config=context.obj,
            experiment_dir=context.obj.get("experiment_dir"),
            trial=trial,
        )
        results.append(result)


@cli.command("amse")
@click.pass_context
def amse(context,):
    workflows.evaluation.amse(
        experiment_dir=Path(context.obj["experiment_dir"]),
        output_dir=Path(context.obj["output_dir"]),
    )


@cli.command("plot-convergence-in-out")
@click.option(
    "--prefix",
    type=str,
    default="baseline",
    help="prefix for the experiment directory",
)
@click.option(
    "--methods", "-m", multiple=True, type=str, help="Which methods to plot",
)
@click.pass_context
def plot_convergence_in_out(
    context, prefix, methods,
):
    workflows.evaluation.plot_convergence_in_out(
        experiment_dir=Path(context.obj["experiment_dir"]),
        prefix=prefix,
        methods=methods,
    )



if __name__ == "__main__":
    print("Is cuda available?", torch.cuda.is_available())
    cli()