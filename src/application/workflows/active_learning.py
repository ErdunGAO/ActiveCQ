import json
from src.library import datasets
from src.library import acquisitions
from src.application.workflows import utils
from pathlib import Path
import time


def active_learner(config, experiment_dir, trial):

    # Set dataset seeds
    dataset_name = config.get("dataset_name")
    config["ds_train"]["seed"] = trial
    config["ds_valid"]["seed"] = trial + 1 if dataset_name == "simulation" else trial
    config["ds_test"]["seed"] = trial + 2 if dataset_name == "simulation" else trial

    # Get datasets
    ds_active = datasets.ActiveLearningDataset(
        datasets.DATASETS.get(dataset_name)(**config.get("ds_train"))
    )

    # Update the config after generating the dataset
    values = {
    "condition_value": ds_active.dataset.condition_value,
    "treatment_value": ds_active.dataset.treatment_value,
    "treatment_interest": ds_active.dataset.treatment_interest
    }
    for key, value in values.items():
        config[key] = value
        for ds in ["train", "valid", "test"]:
            config[f"ds_{ds}"][key] = value

    ds_valid = datasets.DATASETS.get(dataset_name)(**config.get("ds_valid"))

    # set the seed
    utils.set_seed(config.get("seed"))

    # Set the trial dir
    acquisition_function_name = config.get("acquisition_function")
    if "acqe" in acquisition_function_name:
        method_name = f"{config.get('adaptive_strategy')}_{config.get('cde_estimator')}_{config.get('batch_aware')}"
    else:
        method_name = acquisition_function_name
    trial_dir = Path(experiment_dir) / method_name / f"trial-{trial:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    # Write config for downstream use
    config_path = trial_dir / "config.json"
    with config_path.open(mode="w") as cp:
        json.dump(config, cp)

    # Get the acquisition function

    acquisition_function = acquisitions.FUNCTIONS.get(
        config.get("acquisition_function")
    )

    # Do active learning loop
    acq_size = config.get("acq_size")
    warm_start_size = config.get("warm_start_size")
    max_acquisitions = config.get("max_acquisitions")
    condition_type = config.get("condition_type")
    trained_model = None

    # Loop over acquisitions
    acquisition_times = []
    for i in range(max_acquisitions):
        acquisition_dir = trial_dir / f"acquisition-{i:03d}"
        acquired_path = acquisition_dir / "aquired.json"
        acquisition_dir.mkdir(parents=True, exist_ok=True)

        # Get the new data points
        if i == 0:
            cde = None
            idx_list, cde = acquisitions.random(model=trained_model,
                                    active_dataset=ds_active,
                                    acq_size=warm_start_size,
                                    condition_type=condition_type,
                                    cde=cde)
        else:
            # Predict pool set
            start_time = time.time()
            assert trained_model is not None, "Model is not trained"
            idx_list, cde = acquisition_function(model=trained_model,
                                    active_dataset=ds_active,
                                    acq_size=acq_size,
                                    condition_type=condition_type,
                                    config=config,
                                    cde=cde)

            end_time = time.time()
            acquisition_times.append(end_time - start_time)
        
        ds_active.acquire(idx_list)

        # Train the model
        trained_model = utils.train_imp(
            ds_train=ds_active,
            ds_valid=ds_valid,
            job_dir=acquisition_dir,
            config=config,
        )
        
        
        # Save acuired points
        with acquired_path.open(mode="w") as ap:
            json.dump(
                {"aquired_indices": [int(a) for a in ds_active.acquired_indices]},
                ap,
            )
    
    # Save acquisition times
    average_time = sum(acquisition_times) / len(acquisition_times)
    acquisition_times_path = trial_dir / "acquisition_times.json"
    with acquisition_times_path.open(mode="w") as atp:
        json.dump({"acquisition_times": acquisition_times, "average_time": average_time}, atp)