import json
import os
import copy

import torch
import numpy as np
from scipy import stats

from torch.utils import data

from src.library import datasets
from src.library import acquisitions
from src.application.workflows import utils

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# sns.set(style="whitegrid", palette="colorblind")
params = {
    "figure.constrained_layout.use": True,
    "axes.labelsize": 20,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "axes.titlesize": 25,
    "legend.fontsize": 14,
    "legend.title_fontsize": 14,
    "lines.markersize": 8,
    # "font.size": 24,
    "axes.grid": True,         
    "grid.color": "gray",     
    "grid.linestyle": "--",   
    "grid.linewidth": 0.5,  
    "grid.alpha": 0.7,
}
plt.rcParams.update(params)

styles = {
    "Random": ("C0", "--", "o"),  # Default color (blue) with dashed line
    r"$\mu$-BALD": ("C1", "--", "o"),  # Default color (orange) with dashed line
    r"$\mu$-BALD-S": ("C1", "--", "s"),  # Default color (orange) with dashed line
    "Coresets": ("C7", "--", "o"),  # Default color (green) with dashed line
    "TVR": ("C3", "--", "o"),  # Default color (green) with dashed line
    # "distance_rank": ("C4", "--", "o"),  # Default color (green) with dashed line
    ##### New methods #####
    "IG_MDN": ("C5", "-", "X"),  # Default color (green) with dashed line
    "IG_MDN_G": ("C5", "-", "D"),  # Default color (green) with dashed line
    "IG_CME": ("C6", "-", "X"),  # Default color (green) with dashed line
    "IG_CME_G": ("C6", "-", "D"),  # Default color (green) with dashed line
    "TVR_MDN": ("C2", "-", "X"),  # Default color (green) with dashed line
    "TVR_MDN_G": ("C2", "-", "D"),  # Default color (green) with dashed line
    "TVR_CME": ("#FF5733", "-", "X"),  # Default color (green) with dashed line
    "TVR_CME_G": ("#FF5733", "-", "D"),  # Default color (green) with dashed line
    "TVR_CME_S": ("#FF5733", "-", "s"),  # Default color (green) with dashed line
    "TVR_MDN_S": ("C2", "-", "s"),  # Default color (green) with dashed line
    "IG_CME_S": ("C6", "-", "s"),  # Default color (green) with dashed line
    "IG_MDN_S": ("C5", "-", "s"),  # Default color (green) with dashed line


}


# New implementation of our case
def amse(experiment_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    amse = {}
    trial = 0
    for trial_dir in sorted(experiment_dir.iterdir()):
        if "trial-" not in str(trial_dir):
            continue
        trial_key = f"trial-{trial:03d}"
        config_path = trial_dir / "config.json"
        with config_path.open(mode="r") as cp:
            config = json.load(cp)

        dataset_name = config.get("dataset_name")

        config["ds_train"]["seed"] = trial
        config["ds_valid"]["seed"] = trial + 1 if dataset_name == "simulation" else trial
        config["ds_test"]["seed"] = trial + 2 if dataset_name == "simulation" else trial

        # Get datasets
        ds_valid = datasets.DATASETS.get(dataset_name)(**config.get("ds_valid"))
        ds_test_in_config = config["ds_train"].copy()
        ds_test_in_config["training_mode"] = False
        ds_test_in = datasets.DATASETS.get(dataset_name)(**ds_test_in_config)
        ds_test_out = datasets.DATASETS.get(dataset_name)(**config.get("ds_test"))

        acquisition_function = acquisitions.FUNCTIONS.get(
            config.get("acquisition_function")
        )

        trial_amse_path = trial_dir / "amse.json"
        if os.path.isfile(trial_amse_path):
            os.remove(trial_amse_path)
            print(f"Deleted: {trial_amse_path}")
        
        if not trial_amse_path.exists():
            trial_amse = {"value_in": [], "value_out": [], "num_acquired": [], "time": []}

            # Load the model
            max_acquisitions = config.get("max_acquisitions")
            for i in range(max_acquisitions):
                # Load acquired indices
                acquired_path = trial_dir / f"acquisition-{i:03d}" / "aquired.json"
                if not acquired_path.exists():
                    break
                with acquired_path.open(mode="r") as ap:
                    aquired_dict = json.load(ap)
                acquired_indices = aquired_dict["aquired_indices"]
                num_acquired = len(acquired_indices)
                acquisition_dir = trial_dir / f"acquisition-{i:03d}"
                # mu_0, mu_1 = utils.PREDICT_FUNCTIONS[config.get("model_name")](
                #     dataset=ds_test, job_dir=acquisition_dir, config=config
                # )
                #RESUME here
                #NOTE: (BE CAREFUL) the new model should be initialized with the same training data as the previous model
                # refer to https://github.com/cornellius-gp/gpytorch/issues/677
                ds_active = datasets.ActiveLearningDataset(
                        datasets.DATASETS.get(dataset_name)(**config.get("ds_train"))
                    )
                ds_active.acquire(acquired_indices)
                amse_in = utils.predict_imp(
                    ds_train=ds_active, ds_valid=ds_valid, ds_test=ds_test_in, job_dir=acquisition_dir, config=config
                )

                amse_out = utils.predict_imp(
                    ds_train=ds_active, ds_valid=ds_valid, ds_test=ds_test_out, job_dir=acquisition_dir, config=config
                )

                trial_amse["value_in"].append(float(amse_in))
                trial_amse["value_out"].append(float(amse_out))
                trial_amse["num_acquired"].append(num_acquired)

            # get the time
            acquisition_times_path = trial_dir / "acquisition_times.json"
            with acquisition_times_path.open(mode="r") as atp:
                data = json.load(atp)
            acquisition_times = data.get("acquisition_times", [])
            trial_amse["time"].append(acquisition_times)
            trial_amse_path.write_text(json.dumps(trial_amse, indent=4, sort_keys=True))
        else:
            trial_amse = json.loads(trial_amse_path.read_text())
        amse[trial_key] = trial_amse
        trial += 1
    amses_in = []
    amses_out = []
    num_acquired = []
    times = []
    for trial, results in amse.items():
        amses_in.append(results["value_in"])
        amses_out.append(results["value_out"])
        num_acquired.append(results["num_acquired"])
        times.append(results["time"])

    amses_in = np.asarray(amses_in)
    amses_out = np.asarray(amses_out)
    print(num_acquired[0])
    print(list(amses_in.mean(0)))
    print(list(stats.sem(amses_in, axis=0)))
    print(list(amses_out.mean(0)))
    print(list(stats.sem(amses_out, axis=0)))
    overall_time_mean= np.mean(times)
    overall_time_std = np.std(times)
    print(f"Overall time mean: {overall_time_mean}")
    print(f"Overall time std: {overall_time_std}")

    acquisition_function = config.get("acquisition_function")
    if "acqe" in acquisition_function:
        method_name = f"{config.get('adaptive_strategy')}_{config.get('cde_estimator')}_{config.get('batch_aware')}"
    else:
        method_name = acquisition_function

    acquisition_function = method_name
    amse["acquisition_function"] = acquisition_function
    amse["task_type"] = config.get("task_type")
    amse["overall_time_mean"] = overall_time_mean
    amse["overall_time_std"] = overall_time_std
    amse_path = output_dir / f"{acquisition_function}_amse.json"
    amse_path.write_text(json.dumps(amse, indent=4, sort_keys=True))


def get_label_name(acquisition_function):
    if acquisition_function == "var_rank":
        return r"$\mu$-BALD"
    elif acquisition_function == "random":
        return "Random"
    elif acquisition_function == "coresets":
        return "Coresets"
    elif acquisition_function == "distance_rank":
        return "Dis-Sort"
    elif acquisition_function == "var_reduction_rank":
        return "TVR"
    elif acquisition_function == "var_rank_S":
        return r"$\mu$-BALD-S"
    else:
        if acquisition_function.endswith("_B"):
            acquisition_function = acquisition_function[:-2]
        
        if acquisition_function.startswith("VR"):
            acquisition_function = "TVR" + acquisition_function[2:]
        
        return acquisition_function


def plot_convergence_in_out(experiment_dir, prefix, methods):
    for value in ["value_in", "value_out"]:
        _ = plt.figure(figsize=(7, 6), dpi=500)
        for acquisition_function in methods:
            amse_path = experiment_dir / f"{acquisition_function}_amse.json"
            with amse_path.open("r") as pp:
                original_data = json.load(pp)
                amse_data = copy.deepcopy(original_data)

            task_type = amse_data.pop("task_type", "unknown")
            amse_data.pop("acquisition_function", None)
            amse_data.pop("overall_time_mean", None)
            amse_data.pop("overall_time_std", None)

            amses = []
            num_acquired = None

            for trial_key, results in amse_data.items():
                if not trial_key.startswith("trial-"):
                    continue
                if value not in results:
                    continue
                amses.append(results[value])
                if num_acquired is None:
                    num_acquired = results["num_acquired"]

            amses = np.asarray(amses)
            amses_clean = amses[~np.isnan(amses).any(axis=1)]
            mean_amse = amses_clean.mean(axis=0)
            sem_amse = stats.sem(amses_clean, axis=0)

            # Store mean and sem into the original data structure
            original_data[f"{value}_mean"] = mean_amse.tolist()
            original_data[f"{value}_std"] = sem_amse.tolist()

            # Write back to JSON file
            with amse_path.open("w") as pp:
                json.dump(original_data, pp, indent=2)

            # Plot
            x = np.asarray(num_acquired)
            label_name = get_label_name(acquisition_function)
            _ = plt.plot(
                x,
                mean_amse,
                color=styles[label_name][0],
                linestyle=styles[label_name][1],
                marker=styles[label_name][2],
                markersize=7,
                label=label_name,
            )
            _ = plt.fill_between(
                x=x,
                y1=mean_amse - sem_amse,
                y2=mean_amse + sem_amse,
                color=styles[label_name][0],
                alpha=0.2,
            )
            _ = plt.legend(loc=None, title=None)

        plt.title(f"{task_type.upper()}")
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.5)

        plt.xlabel("Number of acquired observations")
        plt.ylabel(r"$\sqrt{AMSE}$")
        plt.grid(True)
        _ = plt.savefig(experiment_dir / f"{prefix}_{value}_convergence.pdf", dpi=500, bbox_inches='tight')
