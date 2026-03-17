from threading import Condition
from gpytorch.models import gp
import numpy as np
from src.library import models
import torch
import os
import random


def set_seed(seed):
    """Set random seed for reproducibility.

    Args:
        seed (int): Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    try:
        os.environ['PYTHONHASHSEED'] = str(seed)
    except:
        pass


def train_imp(ds_train, ds_valid, job_dir, config):
    # Get model parameters from config
    task_type = config.get("task_type")
    condition_type = config.get("condition_type")
    treatment_type = config.get("treatment_type")
    learning_rate = config.get("learning_rate")
    gp_epochs = config.get("gp_epochs")
    cme_epochs = config.get("cme_epochs")
    exp_mode = config.get("exp_mode")
    learn_cme = config.get("learn_cme")
    if task_type == "cate":
        model = models.IMP_CATE(
            job_dir=job_dir,
            condition_type=condition_type,
            task_type=task_type,
            treatment_type=treatment_type,
            active_dataset=ds_train,
            tune_dataset=ds_valid,
            learning_rate=learning_rate,
            epochs_gp=gp_epochs,
            epochs_cme=cme_epochs,
            learn_cme=learn_cme,
            patience=10,
            num_workers=0,
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            kernel_type=config.get("kernel_type"),
            seed=config.get("seed"),
        )
    elif task_type == "ate":
        model = models.IMP_ATE(
            job_dir=job_dir,
            task_type=task_type,
            treatment_type=treatment_type,
            active_dataset=ds_train,
            tune_dataset=ds_valid,
            learning_rate=learning_rate,
            epochs_gp=gp_epochs,
            patience=10,
            num_workers=0,
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            seed=config.get("seed"),
        )
    elif task_type == "ds":
        model = models.IMP_DS(
            job_dir=job_dir,
            task_type=task_type,
            treatment_type=treatment_type,
            active_dataset=ds_train,
            tune_dataset=ds_valid,
            learning_rate=learning_rate,
            epochs_gp=gp_epochs,
            patience=10,
            num_workers=0,
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            seed=config.get("seed"),
        )
    elif task_type == "scate":
        model = models.IMP_SCATE(
            job_dir=job_dir,
            task_type=task_type,
            treatment_type=treatment_type,
            active_dataset=ds_train,
            tune_dataset=ds_valid,
            learning_rate=learning_rate,
            epochs_gp=gp_epochs,
            epochs_cme=cme_epochs,
            learn_cme=learn_cme,
            patience=10,
            num_workers=0,
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            seed=config.get("seed"),
        )
    elif task_type == "att":
        model = models.IMP_ATT(
            job_dir=job_dir,
            task_type=task_type,
            treatment_type=treatment_type,
            active_dataset=ds_train,
            tune_dataset=ds_valid,
            learning_rate=learning_rate,
            epochs_gp=gp_epochs,
            epochs_cme=cme_epochs,
            learn_cme=learn_cme,
            patience=10,
            num_workers=0,
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            seed=config.get("seed"),
        )
    else:
        raise ValueError(f"Task type {task_type} not supported")
    
    # Train the model
    _ = model.fit(exp_mode=exp_mode)

    return model


def predict_imp(ds_train, ds_valid, ds_test, job_dir, config):

    task_type = config.get("task_type")
    condition_type = config.get("condition_type")
    treatment_type = config.get("treatment_type")
    learning_rate = config.get("learning_rate")
    gp_epochs = config.get("gp_epochs")
    cme_epochs = config.get("cme_epochs")
    learn_cme = config.get("learn_cme")
    if task_type == "cate":
        model = models.IMP_CATE(
            job_dir=job_dir,
            task_type=task_type,
            condition_type=condition_type,
            treatment_type=treatment_type,
            active_dataset=ds_train,
            tune_dataset=ds_valid,
            learning_rate=learning_rate,
            epochs_gp=gp_epochs,
            epochs_cme=cme_epochs,
            learn_cme=learn_cme,
            patience=5,
            num_workers=0,
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            kernel_type=config.get("kernel_type"),
            seed=config.get("seed"),
        )
    elif task_type == "ate":
        model = models.IMP_ATE(
            job_dir=job_dir,
            task_type=task_type,
            treatment_type=treatment_type,
            active_dataset=ds_train,
            tune_dataset=ds_valid,
            learning_rate=learning_rate,
            epochs_gp=gp_epochs,
            patience=5,
            num_workers=0,
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            seed=config.get("seed"),
        )
    elif task_type == "ds":
        model = models.IMP_DS(
            job_dir=job_dir,
            task_type=task_type,
            treatment_type=treatment_type,
            active_dataset=ds_train,
            tune_dataset=ds_valid,
            learning_rate=learning_rate,
            epochs_gp=gp_epochs,
            patience=5,
            num_workers=0,
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            seed=config.get("seed"),
        )
    elif task_type == "scate":
        model = models.IMP_SCATE(
            job_dir=job_dir,
            task_type=task_type,
            treatment_type=treatment_type,
            active_dataset=ds_train,
            tune_dataset=ds_valid,
            learning_rate=learning_rate,
            epochs_gp=gp_epochs,
            epochs_cme=cme_epochs,
            learn_cme=learn_cme,
            patience=5,
            num_workers=0,
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            seed=config.get("seed"),
        )
    elif task_type == "att":
        model = models.IMP_ATT(
            job_dir=job_dir,
            task_type=task_type,
            treatment_type=treatment_type,
            active_dataset=ds_train,
            tune_dataset=ds_valid,
            learning_rate=learning_rate,
            epochs_gp=gp_epochs,
            epochs_cme=cme_epochs,
            learn_cme=learn_cme,
            patience=5,
            num_workers=0,
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            seed=config.get("seed"),
        )
    else:
        raise ValueError(f"Task type {task_type} not supported")
    
    # Load the model
    model.load()
    return model.predict(ds_test)

