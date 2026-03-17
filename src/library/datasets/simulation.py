from enum import unique
from numpy.core.shape_base import atleast_2d
import torch
import numpy as np
from torch.utils import data
from scipy.special import expit as sigmoid
from scipy.stats import norm
from torch.utils.data.dataset import T
import warnings

def fix_values_int(arr, step=0.2):
    scale = int(1 / step)
    scaled = np.round(arr * scale).astype(int) 
    result = scaled / scale 
    return np.round(result, decimals=10) 

class Simulation(data.Dataset):
    def __init__(
        self,
        num_examples,
        treatment_type="binary", # binary or continuous
        training_mode = True,
        task_type="cate",  # ate, cate, att, scate
        dim_adjustment=4,
        dataset_type="train", # train, valid, test
        condition_type="discrete", # discrete, continuous
        condition_value=None,
        treatment_value=None,
        treatment_interest=None,
        treatment_assignment="weak", # weak, strong
        regression_func="easy",
        condition_dist="easy",
        seed=2024,
    ):
        super(Simulation, self).__init__()
        rng = np.random.RandomState(seed=seed)
        self.num_examples = num_examples
        self.treatment_type = treatment_type
        self.dim_adjustment = dim_adjustment #NOTE: useless for now, 19-Dec-2024
        self.dataset_type = dataset_type
        self.condition_value = condition_value
        self.condition_type = condition_type
        self.treatment_value = treatment_value
        self.treatment_interest = treatment_interest
        self.dim_condition = 1
        self.dim_treatment = 1
        self.dim_output = 1
        self.regression_func = regression_func
        self.treatment_assignment = treatment_assignment
        self.condition_dist = condition_dist
        self.task_type = task_type
        self.seed = seed

        # Security check
        assert not (task_type == "att" and treatment_type == "continuous"), "ATT is not defined for continuous treatment"
        assert self.condition_type == "continuous", "Only continuous condition is supported in the simulation, for now"
        assert task_type == "att" or self.treatment_value is None, "Treatment value is only allowed for ATT"
        
        # Detailed settings for different tasks
        if self.dataset_type == "train" and training_mode:
            if task_type in ["cate", "scate"]:
                if self.condition_value is None:
                    if self.condition_dist == "easy":
                        self.condition_value = rng.uniform(-0.5, 0.5)
                    elif self.condition_dist == "hard":
                        self.condition_value = rng.uniform(-2, 2)
                    else:
                        raise NotImplementedError(
                            f"{condition_dist} not supported. Choose from 'easy' or 'hard'"
                        )

            if task_type == "att":
                if self.treatment_value is None:
                    if self.treatment_type == "discrete":
                        self.treatment_value = rng.choice([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
                    elif self.treatment_type == "binary":
                        self.treatment_value = rng.choice([0.0,1.0])
                    else:
                        raise NotImplementedError(
                            f"{treatment_type} not supported. Choose from 'binary' or 'discrete'"
                        )

            if self.treatment_interest:
                assert self.treatment_type in ["discrete", "binary"], "If you set 'treatment_interest' as True, meaning you want to \
                    run the 'interest_shift' setup. In this case, the treatment type should be either 'discrete' or 'binary'. If it is 'continuous', \
                        it's hard to generate the test dataset with the same treatment interest."
                
                if self.treatment_type == "discrete":
                    lst = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
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
        
        
        # Generate the condition variable
        if self.condition_dist == "easy":
            self.condition = rng.uniform(-0.5, 0.5, size=(num_examples, 1)).astype("float32")
        elif self.condition_dist == "hard":
            self.condition = rng.uniform(-2, 2, size=(num_examples, 1)).astype("float32")
        else:
            raise NotImplementedError(
                f"{condition_dist} not supported. Choose from 'easy' or 'hard'"
            )

        # For the testing dataset
        if dataset_type == "test" and task_type in ["cate", "scate"]:
            self.condition = np.full_like(self.condition, condition_value, dtype="float32")
        
        if dataset_type == "train" and task_type in ["cate", "scate"]:
            if not training_mode:
                self.condition = np.full_like(self.condition, condition_value, dtype="float32")


        ############################################################################################################
        # Generate the adjustment variable
        ############################################################################################################

        # This case is with fixed data generation process
        if self.condition_dist == "easy":
            s1 = 1 + 2 * self.condition + rng.uniform(-0.5, 0.5, size=(num_examples, 1)).astype("float32")
            s2 = 1 + 2 * self.condition + rng.uniform(-0.5, 0.5, size=(num_examples, 1)).astype("float32")
            s3 = (self.condition - 1) ** 2 + rng.uniform(-0.5, 0.5, size=(num_examples, 1)).astype("float32")

        elif self.condition_dist == "hard":
            s1 = np.cos(self.condition) + self.condition + rng.standard_normal(size=(num_examples, 1))
            s2 = -1 + 0.25 * self.condition * self.condition + rng.standard_normal(size=(num_examples, 1))
            s3 = np.sin(self.condition) ** 2 + rng.standard_normal(size=(num_examples, 1))

        else:
            raise NotImplementedError(
                f"{condition_dist} not supported. Choose from 'easy' or 'hard'"
            )

        if task_type == "cate":
            inner = 2*rng.standard_normal(size=(num_examples, 1))
            s4 = np.exp(inner) + rng.standard_normal(size=(num_examples, 1))
            self.adjustment = np.hstack([s1, s2, s3, s4])
        else:
            self.adjustment = np.hstack([s1, s2, s3])
        
        self.dim_adjustment = self.adjustment.shape[1] #Hard check


        if task_type == "ds":
            ds_condition = rng.uniform(-1, 1, size=(num_examples, 1)).astype("float32")
            s1_ds = rng.uniform(-1, 1, size=(num_examples, 1)).astype("float32")
            s2_ds = rng.uniform(-0.5, 0, size=(num_examples, 1)).astype("float32")
            s3_ds = rng.uniform(0, 0.5, size=(num_examples, 1)).astype("float32")
            self.ds_condition = ds_condition
            self.ds_adjustment = np.hstack([s1_ds, s2_ds, s3_ds])
            X_ds = np.hstack([self.ds_condition, s1_ds, s2_ds, s3_ds])
        
        if treatment_type == "binary":
            if self.treatment_assignment == "weak":
                X = np.hstack([self.condition, s1, s2, s3])
                theta = np.array([1.0 / ((i + 1) ** 2) for i in range(X.shape[1])])
                treatment_org = norm.cdf(X.dot(theta) * 3).reshape(-1,1) + 1.5*rng.standard_normal(size=(num_examples, 1)) - 0.5

                if task_type == "ds":
                    X_ds = np.hstack([self.ds_condition, s1_ds, s2_ds, s3_ds])
                    ds_treatment_org = norm.cdf(X_ds.dot(theta) * 3).reshape(-1,1) + 1.5*rng.standard_normal(size=(num_examples, 1)) - 0.5

            elif self.treatment_assignment == "strong":
                X = np.hstack([self.condition, s1, s2, s3])
                theta = np.array([1.0 / ((i + 1) ** 2) for i in range(X.shape[1])])
                treatment_org = norm.cdf(X.dot(theta)).reshape(-1,1) -0.5+ 0.75*rng.standard_normal(size=(num_examples, 1)).astype("float32")
            else:
                raise NotImplementedError(
                    f"{treatment_assignment} not supported. Choose from 'weak' or 'strong'"
                )
            
            self.treatment = np.zeros_like(treatment_org)
            self.treatment[treatment_org > 0] = 1.0

            if task_type == "ds":
                self.ds_treatment = np.zeros_like(ds_treatment_org)
                self.ds_treatment[ds_treatment_org > 0] = 1.0

        elif treatment_type in ["continuous", "discrete"]:
            
            if self.treatment_assignment == "weak":
                X = np.hstack([self.condition, s1, s2, s3])
                theta = np.array([1.0 / ((i + 1) ** 2) for i in range(X.shape[1])])
                treatment = norm.cdf(X.dot(theta) * 3).reshape(-1,1) + 1.5*rng.standard_normal(size=(num_examples, 1)) - 0.5
                self.treatment = 1.0 / (1.0 + np.exp(-treatment))

                if task_type == "ds":
                    X_ds = np.hstack([self.ds_condition, s1_ds, s2_ds, s3_ds])
                    self.ds_treatment = norm.cdf(X_ds.dot(theta) * 3).reshape(-1,1) + 1.5*rng.standard_normal(size=(num_examples, 1)) - 0.5
                    self.ds_treatment = 1.0 / (1.0 + np.exp(-self.ds_treatment))

            elif self.treatment_assignment == "strong":
                X = np.hstack([self.condition, s1, s2, s3])
                theta = np.array([1.0 / ((i + 1) ** 2) for i in range(X.shape[1])])
                self.treatment = norm.cdf(X.dot(theta) * 3).reshape(-1,1) + 0.75 * rng.standard_normal(size=(num_examples, 1)).astype("float32")
            else:
                raise NotImplementedError(
                    f"{treatment_assignment} not supported. Choose from 'weak' or 'strong'"
                )
            
            if treatment_type == "discrete":
                self.treatment = np.round(self.treatment, 1)
                self.treatment = fix_values_int(self.treatment, step=0.1)
                if task_type == "ds":
                    self.ds_treatment = np.round(self.ds_treatment, 1)
                    self.ds_treatment = fix_values_int(self.ds_treatment, step=0.1)

        else:
            raise NotImplementedError(
                f"{treatment_type} not supported. Choose from 'binary' or 'continuous'"
            )



        # Generate potential outcomes and observed outcome
        scale_po = 2
        bias_po = 5
        if task_type in ["ate", "cate", "att", "ds"]:
            if treatment_type == "binary":
                potential_treatment = np.hstack([np.zeros_like(self.treatment), np.ones_like(self.treatment)])
                if self.regression_func == "hard":
                    self.pos = potential_treatment * self.condition * s1  + s2 * s3 + rng.normal(0.0, 0.16, size=(num_examples, 2)).astype("float32")
                elif self.regression_func == "middle":
                    theta_temp = np.array([1.0 / ((i + 1)) for i in range(4)])
                    self.pos = 1.2 * potential_treatment + 1.2 * X.dot(theta_temp).reshape(-1,1) + potential_treatment ** 2 + potential_treatment * self.condition + potential_treatment * s1 + rng.normal(0.0, 0.16, size=(num_examples, 2)).astype("float32")
                elif self.regression_func == "easy":
                    self.pos = 1.2 * potential_treatment + 1.2 * X.dot(theta).reshape(-1,1) + potential_treatment ** 2 + potential_treatment * self.condition + potential_treatment * s1 + rng.normal(0.0, 0.16, size=(num_examples, 2)).astype("float32")
                else:
                    raise NotImplementedError(
                        f"{regression_func} not supported. Choose from 'easy' or 'hard'"
                    )
                
                self.pos = self.pos * scale_po + bias_po
                self.outcome = self.treatment * self.pos[:, 1].reshape(-1,1) + (1 - self.treatment) * self.pos[:, 0].reshape(-1,1)

                if task_type == "ds":
                    # assert self.regression_func == "middle", "Only middle regression function is supported for DS task"
                    ds_potential_treatment = np.hstack([np.zeros_like(self.ds_treatment), np.ones_like(self.ds_treatment)])
                    if self.regression_func == "hard":
                        self.ds_pos = ds_potential_treatment * self.ds_condition * s1_ds  + s2_ds * s3_ds + rng.normal(0.0, 0.16, size=(num_examples, 2)).astype("float32")
                    elif self.regression_func == "middle":
                        self.ds_pos = 1.2 * ds_potential_treatment + 1.2 * X_ds.dot(theta_temp).reshape(-1,1) + ds_potential_treatment ** 2 + ds_potential_treatment * self.ds_condition + ds_potential_treatment * s1_ds + rng.normal(0.0, 0.16, size=(num_examples, 2)).astype("float32")
                    elif self.regression_func == "easy":
                        self.ds_pos = 1.2 * ds_potential_treatment + 1.2 * X_ds.dot(theta).reshape(-1,1) + ds_potential_treatment ** 2 + ds_potential_treatment * self.ds_condition + ds_potential_treatment * s1_ds + rng.normal(0.0, 0.16, size=(num_examples, 2)).astype("float32")
                    else:
                        raise NotImplementedError(
                            f"{regression_func} not supported. Choose from 'easy' or 'hard'"
                        )
                    self.ds_pos = self.ds_pos * scale_po + bias_po

                    self.ds_outcome = self.ds_treatment * self.ds_pos[:, 1].reshape(-1,1) + (1 - self.ds_treatment) * self.ds_pos[:, 0].reshape(-1,1)

            elif treatment_type in ["continuous", "discrete"]:

                unique_treatment, unique_ind = np.unique(self.treatment, return_index=True)
                potential_treatment = np.tile(unique_treatment.reshape(-1,1), (1, self.treatment.shape[0])).T
                if self.regression_func == "hard":
                    self.pos = potential_treatment * self.condition * s1  + s2 * s3 + rng.normal(0.0, 0.16, size=(num_examples, len(unique_treatment))).astype("float32")
                elif self.regression_func == "middle":
                    theta_temp = np.array([1.0 / ((i + 1)) for i in range(4)])
                    self.pos = 1.2*potential_treatment + 1.2 * X.dot(theta_temp).reshape(-1,1) + potential_treatment** 2 + potential_treatment* self.condition + potential_treatment * s1 + rng.normal(0.0, 0.16, size=(num_examples, len(unique_treatment))).astype("float32")
                elif self.regression_func == "easy":
                    self.pos = 1.2*potential_treatment + 1.2 * X.dot(theta).reshape(-1,1) + potential_treatment** 2 + potential_treatment* self.condition + potential_treatment * s1 + rng.normal(0.0, 0.16, size=(num_examples, len(unique_treatment))).astype("float32")
                else:
                    raise NotImplementedError(
                        f"{regression_func} not supported. Choose from 'easy' or 'hard'"
                    )

                self.pos = self.pos * scale_po + bias_po
                ind_in_unique = np.searchsorted(unique_treatment, self.treatment)
                self.outcome = self.pos[np.arange(num_examples), ind_in_unique.squeeze()].reshape(-1,1)

                if task_type == "ds":
                    # assert self.regression_func == "middle", "Only middle regression function is supported for DS task"
                    ds_unique_treatment, _ = np.unique(self.ds_treatment, return_index=True)
                    ds_potential_treatment = np.tile(ds_unique_treatment.reshape(-1,1), (1, self.ds_treatment.shape[0])).T
                    if self.regression_func == "hard":
                        self.ds_pos = ds_potential_treatment * self.ds_condition * s1_ds  + s2_ds * s3_ds + rng.normal(0.0, 0.16, size=(num_examples, len(ds_unique_treatment))).astype("float32")
                    elif self.regression_func == "middle":
                        self.ds_pos = 1.2*ds_potential_treatment + 1.2 * X_ds.dot(theta_temp).reshape(-1,1) + ds_potential_treatment** 2 + ds_potential_treatment* self.condition + ds_potential_treatment * s1_ds + rng.normal(0.0, 0.16, size=(num_examples, len(ds_unique_treatment))).astype("float32")
                    elif self.regression_func == "easy":
                        self.ds_pos = 1.2*ds_potential_treatment + 1.2 * X_ds.dot(theta).reshape(-1,1) + ds_potential_treatment** 2 + ds_potential_treatment* self.ds_condition + ds_potential_treatment * s1_ds + rng.normal(0.0, 0.16, size=(num_examples, len(ds_unique_treatment))).astype("float32")
                    else:
                        raise NotImplementedError(
                            f"{regression_func} not supported. Choose from 'easy' or 'hard'"
                        )
                    self.ds_pos = self.ds_pos * scale_po + bias_po
                    
                    ds_ind_in_unique = np.searchsorted(ds_unique_treatment, self.ds_treatment)
                    self.ds_outcome = self.ds_pos[np.arange(num_examples), ds_ind_in_unique.squeeze()].reshape(-1,1)

        #NOTE: Maybe we don't need this case, boring
        elif task_type == "scate":
            warnings.warn("The scate task is not updated.")
            if treatment_type == "binary":
                potential_treatment = np.hstack([np.zeros_like(self.treatment), np.ones_like(self.treatment)])
                if self.regression_func == "hard":
                    self.pos = potential_treatment * self.condition * self.condition + np.sin(self.condition) * np.sin(self.condition) + self.condition  + rng.normal(0.0, 0.25, size=(num_examples, 2)).astype("float32")
                elif self.regression_func == "easy":
                    self.pos = 2*potential_treatment * self.condition + rng.normal(0.0, 0.25, size=(num_examples, 2)).astype("float32")
                else:
                    raise NotImplementedError(
                        f"{regression_func} not supported. Choose from 'easy' or 'hard'"
                    )
                self.pos = self.pos * scale_po + bias_po
                self.outcome = self.treatment * self.pos[:, 1].reshape(-1,1) + (1 - self.treatment) * self.pos[:, 0].reshape(-1,1)
            elif treatment_type in ["continuous", "discrete"]:
                unique_treatment, unique_ind = np.unique(self.treatment, return_index=True)
                potential_treatment = np.tile(unique_treatment.reshape(-1,1), (1, self.treatment.shape[0])).T
                if self.regression_func == "hard":
                    self.pos = potential_treatment * self.condition * self.condition + np.sin(self.condition) * np.sin(self.condition) + self.condition  + rng.normal(0.0, 0.25, size=(num_examples, len(unique_treatment))).astype("float32")
                elif self.regression_func == "easy":
                    self.pos = 2*potential_treatment * self.condition + rng.normal(0.0, 0.25, size=(num_examples, len(unique_treatment))).astype("float32")
                else:
                    raise NotImplementedError(
                        f"{regression_func} not supported. Choose from 'easy' or 'hard'"
                    )
                self.pos = self.pos * scale_po + bias_po
                ind_in_unique = np.searchsorted(unique_treatment, self.treatment)
                self.outcome = self.pos[np.arange(num_examples), ind_in_unique.squeeze()].reshape(-1,1)
            else:
                raise NotImplementedError(
                    f"{treatment_type} not supported. Choose from 'binary' or 'continuous' or 'discrete'"
                )

        else:
            raise NotImplementedError(
                f"{task_type} not supported. Choose from 'ate', 'cate', 'att' or 'scate'"
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
            if task_type in ["ate", "cate", "scate"]:
                self.inputs_set = {"condition": np.tile(self.condition, (potential_treatment.shape[1],1)),
                            "treatment": potential_treatment.T.reshape(-1, 1),
                            "adjustment": np.tile(self.adjustment, (potential_treatment.shape[1],1))}
                self.targets = self.pos.T.reshape(-1, 1)
            elif task_type == "ds":
                self.inputs_set = {"condition": np.tile(self.condition, (potential_treatment.shape[1],1)),
                            "treatment": potential_treatment.T.reshape(-1, 1),
                            "adjustment": np.tile(self.adjustment, (potential_treatment.shape[1],1))}
                self.targets = self.pos.T.reshape(-1, 1)
                
                self.inputs_set["ds_treatment"] = potential_treatment.T.reshape(-1, 1)
                self.inputs_set["ds_adjustment"] = np.tile(self.ds_adjustment, (potential_treatment.shape[1],1))
                self.inputs_set["ds_condition"] = np.tile(self.ds_condition, (potential_treatment.shape[1],1))

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
                    f"{task_type} not supported. Choose from 'ate', 'cate', 'att' or 'scate'"
                )
        else:
            raise NotImplementedError(
                f"{dataset_type} not supported. Choose from 'train', 'valid' or 'test'"
            )

        #NOTE: (BE CAREFUL) Please note the order of elements in the inputs_set
        # self.inputs = np.hstack([self.treatment, self.condition, self.adjustment])
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