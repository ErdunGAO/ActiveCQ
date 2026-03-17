import torch
import pyreadr
import requests
import numpy as np

from pathlib import Path

from torch.utils import data
import copy
from sklearn import preprocessing
from sklearn import model_selection

from scipy.stats import norm

def fix_values_int(arr, step=0.2):
    scale = int(1 / step)
    scaled = np.round(arr * scale).astype(int) 
    result = scaled / scale 
    return np.round(result, decimals=10) 

_CONTINUOUS_COVARIATES = [
    "bw",
    "b.head",
    "preterm",
    "birth.o",
    "nnhealth",
    "momage",
]

_BINARY_COVARIATES = [
    "sex",
    "twin",
    "b.marr",
    "mom.lths",
    "mom.hs",
    "mom.scoll",
    "cig",
    "first",
    "booze",
    "drugs",
    "work.dur",
    "prenatal",
    "ark",
    "ein",
    "har",
    "mia",
    "pen",
    "tex",
    "was",
]

_TREATMENT = ["treat"]


class IHDP(data.Dataset):
    def __init__(self,
        root,
        dataset_type="train",
        training_mode = True,
        treatment_type="binary", # binary or continuous
        task_type="cate",  # ate, cate, att
        condition_name="bw",
        condition_type="binary",
        condition_value=None,
        treatment_value=None,
        treatment_interest=None,
        treatment_assignment="weak", # weak, strong
        regression_func="easy",
        seed=2024):

        # Generate response surfaces
        rng = np.random.default_rng(seed)
        self.treatment_type = treatment_type
        self.condition_value = condition_value
        self.condition_name = condition_name
        self.condition_type = condition_type
        self.treatment_value = treatment_value
        self.treatment_interest = treatment_interest
        self.condition_type = "discrete"
        self.dim_adjustment = 24
        self.dim_condition = 1
        self.dim_treatment = 1
        self.dim_output = 1
        self.regression_func = regression_func
        self.treatment_assignment = treatment_assignment
        self.task_type = task_type


        # Detailed settings for different tasks
        if dataset_type == "train" and training_mode:
            if task_type in ["cate", "scate"]:
                if self.condition_value is None:
                    self.condition_value = rng.choice([0.2, 0.4, 0.6, 0.8, 1.0])


            if task_type == "att":
                if self.treatment_value is None:
                    if self.treatment_type == "discrete":
                        self.treatment_value = rng.choice([0.2, 0.4, 0.6, 0.8, 1.0])
                    elif self.treatment_type == "binary":
                        self.treatment_value = rng.choice([0.0, 1.0])
                    else:
                        raise NotImplementedError(
                            f"{treatment_type} not supported. Choose from 'binary' or 'discrete'"
                        )

            if self.treatment_interest:
                assert self.treatment_type in ["discrete", "binary"], "If you set 'treatment_interest' as True, meaning you want to \
                    run the 'interest_shift' setup. In this case, the treatment type should be either 'discrete' or 'binary'. If it is 'continuous', \
                        it's hard to generate the test dataset with the same treatment interest."
                if self.treatment_type == "discrete":
                    lst = [0.2, 0.4, 0.6, 0.8, 1.0]
                    if task_type == "att":
                        lst.remove(self.treatment_value)
                        self.treatment_interest = rng.choice(lst)
                    else:
                        self.treatment_interest = rng.choice(lst)
                elif self.treatment_type == "binary":
                    lst = [0.0, 1.0]
                    if task_type == "att":
                        self.treatment_interest = 1.0 - self.treatment_value
                    else:
                        self.treatment_interest = rng.choice(lst)
                else:
                    raise NotImplementedError(
                        f"{treatment_type} not supported. Choose from 'binary' or 'discrete'"
                    )
            else:
                self.treatment_interest = None
        
        else:
            if task_type in ["cate", "scate"]:
                assert self.condition_value is not None, "Condition value should be set for validation and test dataset"


        ## Sharing part for both the continuous and binary case.
        root = Path(root)
        data_path = root / "ihdp.RData"
        # Download data if necessary
        if not data_path.exists():
            root.mkdir(parents=True, exist_ok=True)
            r = requests.get(
                "https://github.com/vdorie/npci/raw/master/examples/ihdp_sim/data/ihdp.RData"
            )
            with open(data_path, "wb") as f:
                f.write(r.content)
        
        df = pyreadr.read_r(str(data_path))["ihdp"]
        # Make observational as per Hill 2011
        df = df[~((df["treat"] == 1) & (df["momwhite"] == 0))]
        df = df[_CONTINUOUS_COVARIATES + _BINARY_COVARIATES + _TREATMENT]
        # Standardize continuous covariates
        df[_CONTINUOUS_COVARIATES] = preprocessing.StandardScaler().fit_transform(
            df[_CONTINUOUS_COVARIATES]
        )

        for col in _CONTINUOUS_COVARIATES + _BINARY_COVARIATES:
            minval = df[col].min() 
            maxval = df[col].max() 
            if maxval != minval:
                df[col] = (df[col] - minval) / (maxval - minval)
        
        # Split the dataset
        df_train, df_test = model_selection.train_test_split(
                df, test_size=0.2, random_state=seed
            )
        df_train, df_valid = model_selection.train_test_split(
            df_train, test_size=0.2, random_state=seed
            )

        if dataset_type == "train":
            df = df_train
        elif dataset_type == "valid":
            df = df_valid
        elif dataset_type == "test":
            df = df_test
        else:
            raise NotImplementedError(
            f"{dataset_type} not supported. Choose from 'train', 'valid' or 'test'"
        )

        _CONTINUOUS_COVARIATES_copy = _CONTINUOUS_COVARIATES.copy()
        _CONTINUOUS_COVARIATES_copy.remove(self.condition_name)
        adjustment_name_list = _CONTINUOUS_COVARIATES_copy + _BINARY_COVARIATES
        self.adjustment = df[adjustment_name_list].to_numpy(dtype="float32")
        self.condition = df[self.condition_name].to_numpy(dtype="float32").reshape(-1,1)

        # process the condition
        if self.condition_type == "discrete" and task_type == "cate":
            step=0.2
            self.condition = np.round(self.condition / step) * step
            self.condition = np.round(self.condition, decimals=10)
            self.condition[self.condition == 0] = 0.2
            self.condition = fix_values_int(self.condition, step=step)

        num_examples = len(df)

        if task_type == "ds":
            ds_condition = rng.uniform(0, 0.5, size=(num_examples, 1)).astype("float32")
            s1_ds = rng.uniform(0, 0.5, size=(num_examples, 1)).astype("float32")
            s2_ds = rng.uniform(0, 0.5, size=(num_examples, 1)).astype("float32")
            s3_ds = rng.uniform(0, 0.5, size=(num_examples, 1)).astype("float32")
            s4_ds = rng.uniform(0, 0.5, size=(num_examples, 1)).astype("float32")
            s5_ds = rng.uniform(0, 0.5, size=(num_examples, 1)).astype("float32")
            self.ds_condition = ds_condition
            other_adjustment = self.adjustment[:,5:]
            self.ds_adjustment = np.hstack([s1_ds, s2_ds, s3_ds, s4_ds, s5_ds, other_adjustment])

            X_ds = np.hstack([self.ds_condition, s1_ds, s2_ds, s3_ds, s4_ds, s5_ds, other_adjustment])


        # Binary case
        if self.treatment_type == "binary":
            
            t = df["treat"]
            X = df[_CONTINUOUS_COVARIATES + _BINARY_COVARIATES].to_numpy(dtype="float32")
            X_c = X[:,:6]
            theta = np.array([1.0 / (i + 1) for i in range(6)])
            # theta = rng.choice(
            # [0.1, 0.2, 0.3, 0.4],
            # size=(len(_CONTINUOUS_COVARIATES),),
            # p=[0.25, 0.25, 0.25, 0.25],
            # )
            scale_po = 1
            bias_po = 1
            df["y0"] = 1.2 * X_c.dot(theta).reshape(-1,1) + rng.normal(0.0, 0.16, size=t.shape).reshape(-1,1)
            df["y1"] = 1.2 * X_c.dot(theta).reshape(-1,1) + np.exp(X_c +0.5).dot(theta).reshape(-1,1) + 3*X_c[:,0].reshape(-1,1) + rng.normal(0.0, 0.16, size=t.shape).reshape(-1,1)
            df["y0"] = scale_po * df["y0"] + bias_po
            df["y1"] = scale_po * df["y1"] + bias_po
            y = t * df["y1"] + (1 - t) * df["y0"]
            df["y"] = y

            self.treatment = df["treat"].to_numpy(dtype="float32").reshape(-1,1)
            potential_treatment = np.hstack([np.zeros_like(self.treatment).reshape(-1,1), np.ones_like(self.treatment).reshape(-1,1)])
            self.pos = np.hstack([df["y0"].to_numpy(dtype="float32").reshape(-1,1), df["y1"].to_numpy(dtype="float32").reshape(-1,1),])
            self.outcome = df["y"].to_numpy(dtype="float32").reshape(-1,1)

            if task_type == "ds":
                ds_potential_treatment = np.hstack([np.zeros_like(self.treatment).reshape(-1,1), np.ones_like(self.treatment).reshape(-1,1)])
                ds_X_c = X_ds[:,:6]
                ds_theta = np.array([1.0 / (i + 1) for i in range(6)]
                )

                df["ds_y0"] = 1.2 * ds_X_c.dot(ds_theta).reshape(-1,1) + rng.normal(0.0, 0.16, size=t.shape).reshape(-1,1)
                df["ds_y1"] = 1.2 * ds_X_c.dot(ds_theta).reshape(-1,1) + np.exp(ds_X_c +0.5).dot(ds_theta).reshape(-1,1) + 3*ds_X_c[:,0].reshape(-1,1) + rng.normal(0.0, 0.16, size=t.shape).reshape(-1,1)
                df["ds_y0"] = scale_po * df["ds_y0"] + bias_po
                df["ds_y1"] = scale_po * df["ds_y1"] + bias_po

                self.ds_treatment = df["treat"].to_numpy(dtype="float32").reshape(-1,1) #useless
                ds_potential_treatment = np.hstack([np.zeros_like(self.treatment).reshape(-1,1), np.ones_like(self.treatment).reshape(-1,1)])
                self.ds_pos = np.hstack([df["ds_y0"].to_numpy(dtype="float32").reshape(-1,1), df["ds_y1"].to_numpy(dtype="float32").reshape(-1,1),])
                self.ds_outcome = df["y"].to_numpy(dtype="float32").reshape(-1,1) #useless

            

        elif self.treatment_type in ["discrete", "continuous"]:
            
            X = df[_CONTINUOUS_COVARIATES + _BINARY_COVARIATES].to_numpy(dtype="float32")
            cate_idx1 = np.array([3,6,7,8,9,10,11,12,13,14])
            cate_idx2 = np.array([15,16,17,18,19,20,21,22,23,24])

            alpha = 5.
            cate_mean1 = np.mean(np.mean(X[:, cate_idx1], axis=1))
            cate_mean2 = np.mean(np.mean(X[:, cate_idx2], axis=1))
            tem = np.tanh((np.sum(X[:, cate_idx2], axis=1) / 10.0 - cate_mean2) * alpha)

            if self.treatment_assignment == "weak":
                
                X_c = X[:,:6]
                # theta_t = np.array([1.0 / ((i + 1) ** 2) for i in range(6)])
                theta_t = np.array([1.0 / (i + 1) for i in range(6)])
                # theta = rng.choice(
                # [0.1, 0.2, 0.3, 0.4],
                # size=(len(_CONTINUOUS_COVARIATES),),
                # p=[0.25, 0.25, 0.25, 0.25],
                # )
                treatment = norm.cdf(X_c.dot(theta_t) * 3).reshape(-1,1) + 1.5*rng.standard_normal(size=(num_examples, 1)) - 0.5
                self.treatment = 1.0 / (1.0 + np.exp(-treatment))

                if task_type == "ds":
                    ds_X_c = X_ds[:,:6]
                    ds_theta_t = np.array([1.0 / (i + 1) for i in range(6)])
                    ds_treatment = norm.cdf(ds_X_c.dot(ds_theta_t) * 3).reshape(-1,1) + 1.5*rng.standard_normal(size=(num_examples, 1)) - 0.5
                    self.ds_treatment = 1.0 / (1.0 + np.exp(-ds_treatment))
            
            elif self.treatment_assignment == "strong":
                

                x1 = X[:, 0]
                x2 = X[:, 1]
                x3 = X[:, 2]
                x4 = X[:, 3]
                x5 = X[:, 4]

                term1 = x1 / (1.0 + x2)
                term2 = np.max([x3, x4, x5], axis=0) / (0.2 + np.min([x3, x4, x5], axis=0))
                term3 = np.tanh((np.sum(X[:, cate_idx2], axis=1) / 10.0 - cate_mean2) * alpha)

                treatment = term1 + term2 + term3 - 2.0

                self.treatment = 1.0 / (1.0 + np.exp(-2*treatment)).reshape(-1,1)

            else:
                raise NotImplementedError(
                    f"{treatment_assignment} not supported. Choose from 'weak' or 'strong'"
                )
            
            if self.treatment_type == "discrete":
                step=0.2
                self.treatment = np.round(self.treatment/ step) * step
                self.treatment = np.round(self.treatment, decimals=10)
                self.treatment[self.treatment == 0] = 0.2
                self.treatment = fix_values_int(self.treatment, step=step)

                if task_type == "ds":
                    self.ds_treatment = np.round(self.ds_treatment/ step) * step
                    self.ds_treatment = np.round(self.ds_treatment, decimals=10)
                    self.ds_treatment[self.ds_treatment == 0] = 0.2
                    self.ds_treatment = fix_values_int(self.ds_treatment, step=step)



            if self.regression_func == "easy":

                scale_po = 2
                bias_po = 5
                theta = np.array([1.0 / (i + 1) for i in range(6)])
                unique_treatment, _ = np.unique(self.treatment, return_index=True)
                potential_treatment = np.tile(unique_treatment.reshape(-1,1), (1, self.treatment.shape[0])).T
                self.pos = 1.2*potential_treatment + 1.2 * X_c.dot(theta).reshape(-1,1) + self.condition** 2 + potential_treatment* self.condition + rng.normal(0.0, 0.16, size=(num_examples, len(unique_treatment))).astype("float32")
                self.pos = self.pos * scale_po + bias_po

                if task_type == "ds":
                    ds_theta = np.array([1.0 / (i + 1) for i in range(6)])
                    ds_unique_treatment, _ = np.unique(self.ds_treatment, return_index=True)
                    ds_potential_treatment = np.tile(ds_unique_treatment.reshape(-1,1), (1, self.ds_treatment.shape[0])).T
                    self.ds_pos = 1.2*ds_potential_treatment + 1.2 * ds_X_c.dot(ds_theta).reshape(-1,1) + self.ds_condition** 2 + ds_potential_treatment* self.ds_condition + rng.normal(0.0, 0.16, size=(num_examples, len(ds_unique_treatment))).astype("float32")
                    self.ds_pos = self.ds_pos * scale_po + bias_po

                    ds_ind_in_unique = np.searchsorted(ds_unique_treatment, self.ds_treatment)
                    self.ds_outcome = self.ds_pos[np.arange(num_examples), ds_ind_in_unique.squeeze()].reshape(-1,1)
            
            elif self.regression_func == "hard":

                x1 = X[:, 0]
                x2 = X[:, 1]
                x3 = X[:, 2]
                x4 = X[:, 3]
                x5 = X[:, 4]

                # v1
                factor1 = 0.5
                factor2 = 1.5

                # v2
                factor1 = 1.5
                factor2 = 0.5


                scale_po = 1
                bias_po = 1
                unique_treatment, _ = np.unique(self.treatment, return_index=True)
                potential_treatment = np.tile(unique_treatment.reshape(-1,1), (1, self.treatment.shape[0])).T
                self.pos = 1. / (1.2 - potential_treatment) * np.sin(potential_treatment * 3. * 3.14159) * (
                    factor1 * np.tanh((np.sum(X[cate_idx1]) / 10. - cate_mean1) * alpha).reshape(-1,1) +
                    factor2 * (np.exp(0.2 * (x1 - x5)) / (0.1 + np.min(np.stack([x2, x3, x4]), axis=0))).reshape(-1,1) )

            else:
                raise NotImplementedError(
                    f"{regression_func} not supported. Choose from 'easy' or 'hard'"
                )
            ind_in_unique = np.searchsorted(unique_treatment, self.treatment)
            self.outcome = self.pos[np.arange(num_examples), ind_in_unique.squeeze()].reshape(-1,1)

        else:
            raise NotImplementedError(
                    f"{treatment_type} not supported. Choose from 'binary', 'discrete' or 'continuous'"
                )

        # Set the inputs_set and targets
        if ((dataset_type in ["train", "valid"]) and training_mode):
            # for training dataset, very simple
            self.inputs_set = {"condition": self.condition,
                            "treatment": self.treatment,
                            "adjustment": self.adjustment}
            self.targets = self.outcome

            if task_type == "ds":
                self.inputs_set["ds_treatment"] = self.ds_treatment
                self.inputs_set["ds_adjustment"] = self.ds_adjustment
                self.inputs_set["ds_condition"] = self.ds_condition
                self.ds_targets = self.ds_outcome


        elif ((dataset_type == "test") or (not training_mode)):
            # for validation and test dataset, we need to include the potential outcomes
            if task_type in ["ate", "cate"]:
                self.inputs_set = {"condition": np.tile(self.condition, (potential_treatment.shape[1],1)),
                            "treatment": potential_treatment.T.reshape(-1, 1),
                            "adjustment": np.tile(self.adjustment, (potential_treatment.shape[1],1))}
                self.targets = self.pos.T.reshape(-1, 1)
            elif task_type == "ds":
                self.inputs_set = {"condition": np.tile(self.condition, (potential_treatment.shape[1],1)),
                            "treatment": potential_treatment.T.reshape(-1, 1),
                            "adjustment": np.tile(self.adjustment, (potential_treatment.shape[1],1))}
                self.targets = self.pos.T.reshape(-1, 1)

                self.inputs_set["ds_condition"] = np.tile(self.ds_condition, (ds_potential_treatment.shape[1],1))
                self.inputs_set["ds_treatment"] = ds_potential_treatment.T.reshape(-1, 1)
                self.inputs_set["ds_adjustment"] = np.tile(self.ds_adjustment, (ds_potential_treatment.shape[1],1))

                self.ds_targets = self.ds_pos.T.reshape(-1, 1)


            elif task_type == "att":
                # extract the index of treatment group
                idx = np.where(self.treatment == self.treatment_value)[0]
                self.inputs_set = {"condition": np.tile(self.condition[idx], (potential_treatment.shape[1],1)),
                            "treatment": potential_treatment[idx].T.reshape(-1, 1),
                            "adjustment": np.tile(self.adjustment[idx], (potential_treatment.shape[1],1))}
                self.targets = self.pos[idx].T.reshape(-1, 1)
            else:
                raise NotImplementedError(
                    f"{task_type} not supported. Choose from 'ate', 'cate', 'att'"
                )

        self.inputs = np.hstack([self.inputs_set["treatment"],self.inputs_set["condition"], self.inputs_set["adjustment"]]).astype("float32")
        if task_type == "ds":
            self.ds_inputs = np.hstack([self.inputs_set["ds_treatment"],self.inputs_set["ds_condition"], self.inputs_set["ds_adjustment"]]).astype("float32")


        # get unique treatment values
        if self.treatment_interest:
            self.unique_treatment = self.treatment_interest
        else:
            self.unique_treatment = np.unique(self.treatment)



    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]
