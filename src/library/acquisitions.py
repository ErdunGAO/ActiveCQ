from logging import warn
import numpy as np
import copy

from numpy.core.fromnumeric import var
from numpy.random.mtrand import sample

from src.library.modules.cdest.density_estimator.LSCDE import LSConditionalDensityEstimation
from src.library.modules.cdest.density_estimator.KMN import KernelMixtureNetwork
from src.library.modules.cdest.density_estimator.MDN import MixtureDensityNetwork

import random as rand
import warnings

#############################################################
# New acquisition functions
'''
The new acquisition functions.
Inputs:
    model: The model.
    dataset: The dataset.
Returns:
    Idx.
'''
#############################################################

def random(model, active_dataset, acq_size, condition_type, config=None, cde=None):
    # idx = np.random.choice(dataset.pool_dataset.indices, size=dataset.acq_size, replace=False)

    pool_idx = active_dataset.pool_dataset.indices
    assert len(pool_idx) >= acq_size, "The pool size is smaller than the acquisition size."
    idx_list = rand.sample(range(len(pool_idx)), acq_size)

    return idx_list, cde

def coresets(model, active_dataset, acq_size, condition_type, config=None, cde=None):
    # copy the active dataset
    active_dataset = copy.deepcopy(active_dataset)
    selected_idx_fix = active_dataset.training_dataset.indices
    pool_idx_fix = active_dataset.pool_dataset.indices
    assert len(pool_idx_fix) >= acq_size, "The pool size is smaller than the acquisition size."
    idx_list = []

    #ACTIVE
    all_data = active_dataset.dataset.inputs
    _, gamma_matrix = model.compute_bald_variance(all_data)

    # Convert selected_idx_fix and pool_idx_fix to lists for easier manipulation
    selected_indices = selected_idx_fix.tolist()  # Indices of already selected points
    remaining_indices = pool_idx_fix.tolist()  # Indices of points available for selection

    new_selected_indices = [] # are the indices in the all dataset

    # Convert the gamma_matrix to a distance matrix by negating the values
    #NOTE： if the treatment is binary, there would be a distance 1 between different groups. However, it dose not matter.
    distance_matrix = 1-gamma_matrix  # Negate the similarity matrix to get distances

    # Initialize minimum distances for points in the pool to the current selected points
    min_distances = np.full(distance_matrix.shape[0], 1).astype(np.float64)  # Start with a distance of 1, maximum possible distance

    # If we have any initially selected points, update min_distances for the remaining points
    if selected_indices:
        initial_distances = distance_matrix[np.ix_(remaining_indices, selected_indices)]
        min_distances[remaining_indices] = np.min(initial_distances, axis=1)  # Minimize the maximum distance

    # Greedily select points until the acquisition budget is met
    for _ in range(acq_size):
        # Find the index in remaining_indices with the maximum minimum distance to selected points
        max_distance_index = np.argmax(min_distances[remaining_indices]) # Index in remaining_indices, from 0 to len(remaining_indices)
        selected_index = remaining_indices[max_distance_index]  # Absolute index in distance_matrix
        new_selected_indices.append(selected_index)
        selected_indices.append(selected_index)
        remaining_indices.remove(selected_index)

        # Update min_distances for points still in the pool with respect to the new selection
        new_distances = distance_matrix[selected_index, remaining_indices]
        min_distances[remaining_indices] = np.minimum(min_distances[remaining_indices], new_distances)

    # convert the selected index to the pool index
    relative_selected_indices = np.where(np.isin(pool_idx_fix, new_selected_indices))[0]
    idx_list = relative_selected_indices
        
    if model.task_type in ["cate", "scate"]:
        # Analyze the acquired data
        required_idx_in_all_dataset = pool_idx_fix[idx_list]
        selected_condition = active_dataset.dataset.condition[required_idx_in_all_dataset]

        if condition_type == "continuous":
            # get the mean of the absolute value of the selected_condition
            mean_value = np.mean(selected_condition)
            variance_value = np.var(selected_condition)

            print("value mean:", mean_value)
            print("value variance:", variance_value)

        else:
            # Compute the number of each discrete value inside selected_condition
            unique_values, counts = np.unique(selected_condition, return_counts=True)
            # Print each value and its count
            for value, count in zip(unique_values, counts):
                print(f"Value: {value}, Count: {count}")

    return idx_list, cde

def distance_rank(model, active_dataset, acq_size, condition_type, config=None, cde=None):
    # copy the active dataset
    active_dataset = copy.deepcopy(active_dataset)
    pool_idx_fix = active_dataset.pool_dataset.indices
    assert len(pool_idx_fix) >= acq_size, "The pool size is smaller than the acquisition size."
    idx_list = []
    
    condition = active_dataset.dataset.condition[pool_idx_fix]
    var_list = np.abs(condition - active_dataset.dataset.condition_value).squeeze()

    # get the largest-k index
    idx_list = np.argsort(var_list)[:acq_size]
        
    if model.task_type in ["cate", "scate"]:
        # Analyze the acquired data
        required_idx_in_all_dataset = pool_idx_fix[idx_list]
        selected_condition = active_dataset.dataset.condition[required_idx_in_all_dataset]

        if condition_type == "continuous":
            # get the mean of the absolute value of the selected_condition
            mean_value = np.mean(selected_condition)
            variance_value = np.var(selected_condition)

            print("value mean:", mean_value)
            print("value variance:", variance_value)

        else:
            # Compute the number of each discrete value inside selected_condition
            unique_values, counts = np.unique(selected_condition, return_counts=True)
            # Print each value and its count
            for value, count in zip(unique_values, counts):
                print(f"Value: {value}, Count: {count}")

    return idx_list, cde

def var_rank(model, active_dataset, acq_size, condition_type, config=None, cde=None):

    # copy the active dataset
    active_dataset = copy.deepcopy(active_dataset)
    pool_idx_fix = active_dataset.pool_dataset.indices
    assert len(pool_idx_fix) >= acq_size, "The pool size is smaller than the acquisition size."
    idx_list = []

    #ACTIVE
    new_data = active_dataset.dataset.inputs[pool_idx_fix]
    var_list, _ = model.compute_bald_variance(new_data)
    
    # get the largest-k index
    idx_list = np.argsort(var_list)[-acq_size:]
        
    if model.task_type in ["cate", "scate"]:
        # Analyze the acquired data
        required_idx_in_all_dataset = pool_idx_fix[idx_list]
        selected_condition = active_dataset.dataset.condition[required_idx_in_all_dataset]

        if condition_type == "continuous":
            # get the mean of the absolute value of the selected_condition
            mean_value = np.mean(selected_condition)
            variance_value = np.var(selected_condition)

            print("value mean:", mean_value)
            print("value variance:", variance_value)

        else:
            # Compute the number of each discrete value inside selected_condition
            unique_values, counts = np.unique(selected_condition, return_counts=True)
            # Print each value and its count
            for value, count in zip(unique_values, counts):
                print(f"Value: {value}, Count: {count}")

    return idx_list, cde


def var_reduction_rank(model, active_dataset, acq_size, condition_type, config=None, cde=None):
    """
    NOTE: (BE CAREFUL!) In this function, the pool_idx is the true index of the pool dataset in the all dataset.
    idx and idx_list are the index of the pool_idx, not the index of the dataset.
    Inside the active_dataset, the idx would be transformed to the index of the dataset.
    """

    # copy the active dataset
    active_dataset = copy.deepcopy(active_dataset)
    pool_idx_fix = active_dataset.pool_dataset.indices

    # data check
    assert len(pool_idx_fix) >= acq_size, "The pool size is smaller than the acquisition size."

    idx_list = []

    # current data
    temp_data = {
        "treatment": active_dataset.extract_active_data()[0][:,0].reshape(-1,1),
        "condition": active_dataset.extract_active_data()[0][:,1].reshape(-1,1),
        "adjustment": active_dataset.extract_active_data()[0][:,2:],
    }

    var_list = []
    fast_component = {}
    for i in range(len(pool_idx_fix)):
        new_data = {
                    "treatment": active_dataset.dataset.treatment[pool_idx_fix[i]].reshape(1,-1),
                    "condition": active_dataset.dataset.condition[pool_idx_fix[i]].reshape(1,-1),
                    "adjustment": active_dataset.dataset.adjustment[pool_idx_fix[i]].reshape(1,-1),
                }

        var_i, fast_component = model.compute_naive_variance(temp_data, new_data, fast_component, active_dataset)
        var_list.append(var_i)
    
    # get the smallest-k index
    idx_list = np.argsort(var_list)[:acq_size]

    if model.task_type in ["cate", "scate"]:
        # Analyze the acquired data
        required_idx_in_all_dataset = pool_idx_fix[idx_list]
        selected_condition = active_dataset.dataset.condition[required_idx_in_all_dataset]

        if condition_type == "continuous":
            # get the mean of the absolute value of the selected_condition
            mean_value = np.mean(selected_condition)
            variance_value = np.var(selected_condition)

            print(f"Condition value mean: {mean_value}, value variance: {variance_value}")

        else:
            # Compute the number of each discrete value inside selected_condition
            unique_values, counts = np.unique(selected_condition, return_counts=True)
            # Print each value and its count
            for value, count in zip(unique_values, counts):
                print(f"Condition Value: {value}, Count: {count}")
        
        required_idx_in_all_dataset = pool_idx_fix[idx_list]
        selected_treatment = active_dataset.dataset.treatment[required_idx_in_all_dataset]

        unique_values, counts = np.unique(selected_treatment, return_counts=True)
        # Print each value and its count
        for value, count in zip(unique_values, counts):
            print("Treatment Value: ", value, "Count: ", count)
    
    if model.task_type in ["att", "ate"]:
        # Analyze the acquired data
        required_idx_in_all_dataset = pool_idx_fix[idx_list]
        selected_treatment = active_dataset.dataset.treatment[required_idx_in_all_dataset]

        unique_values, counts = np.unique(selected_treatment, return_counts=True)
        # Print each value and its count
        for value, count in zip(unique_values, counts):
            print(f"Value: {value}, Count: {count}")


    return idx_list, cde


#############################################################
#                New acquisition functions                  #
#############################################################

def acqe_run(model, active_dataset, acq_size, cde_estimator, batch_aware, pool_idx_fix, cde, adaptive_strategy):

    idx_list = []
    # current data
    temp_data = {
        "treatment": active_dataset.extract_active_data()[0][:,0].reshape(-1,1),
        "condition": active_dataset.extract_active_data()[0][:,1].reshape(-1,1),
        "adjustment": active_dataset.extract_active_data()[0][:,2:],
    }

    sampling_size = 500
    if active_dataset.dataset.task_type == "cate":
        
        if cde_estimator == "CME":
            pass
        elif cde_estimator in ["MDN", "LSCDE"]:
            all_condition = active_dataset.dataset.condition
            all_adjustment = active_dataset.dataset.adjustment
            dim_adjustment = all_adjustment.shape[1]
            if cde is None:
                if cde_estimator == "MDN":
                    cde = MixtureDensityNetwork(ndim_x=1, ndim_y=dim_adjustment, x_noise_std=0.0001, y_noise_std=0.0001, random_seed=None)
                else:
                    cde = LSConditionalDensityEstimation(ndim_x=1, ndim_y=dim_adjustment,  bandwidth=0.05, random_seed=None)
                cde.fit(all_condition, all_adjustment)
            condition_value = active_dataset.dataset.condition_value
            condition_value_array = np.array([condition_value]*sampling_size).reshape(-1,1)
            sampled_adjustment = cde.sample(condition_value_array)

            # Construct the target data
            condition_value = active_dataset.dataset.condition_value
            unique_treatment = active_dataset.dataset.unique_treatment.reshape(-1,1)
            treatment_expanded = np.repeat(unique_treatment, sampling_size, axis=0)
            condition_expanded = np.full((treatment_expanded.shape[0], 1), condition_value)
            sampled_adjustment_expanded = np.tile(sampled_adjustment, (unique_treatment.shape[0], 1))
            target_data = {
                "treatment": treatment_expanded,
                "condition": condition_expanded,
                "adjustment": sampled_adjustment_expanded,
            }
        else:
            raise ValueError("The cde_estimator should be either 'cme', 'mdn' or 'lscde'.")

    elif active_dataset.dataset.task_type == "ate":
        if cde_estimator == "CME":
            pass
        else:
            adjustment = active_dataset.dataset.adjustment
            condition = active_dataset.dataset.condition
            sampling_size = adjustment.shape[0]
            sampled_adjustment = adjustment

            # Construct the target data
            unique_treatment = active_dataset.dataset.unique_treatment.reshape(-1,1)
            treatment_expanded = np.repeat(unique_treatment, sampling_size, axis=0)
            condition_expanded = np.tile(condition, (unique_treatment.shape[0], 1))
            sampled_adjustment_expanded = np.tile(sampled_adjustment, (unique_treatment.shape[0], 1))
            target_data = {
                "treatment": treatment_expanded,
                "condition": condition_expanded,
                "adjustment": sampled_adjustment_expanded,
            }
    
    elif active_dataset.dataset.task_type == "ds":
        if cde_estimator == "CME":
            pass
        else:
            adjustment = active_dataset.dataset.ds_adjustment
            condition = active_dataset.dataset.ds_condition
            sampling_size = adjustment.shape[0]
            sampled_adjustment = adjustment

            # Construct the target data
            unique_treatment = active_dataset.dataset.unique_treatment.reshape(-1,1)
            treatment_expanded = np.repeat(unique_treatment, sampling_size, axis=0)
            condition_expanded = np.tile(condition, (unique_treatment.shape[0], 1))
            sampled_adjustment_expanded = np.tile(sampled_adjustment, (unique_treatment.shape[0], 1))
            target_data = {
                "treatment": treatment_expanded,
                "condition": condition_expanded,
                "adjustment": sampled_adjustment_expanded,
            }
    
    elif active_dataset.dataset.task_type == "att":
        if cde_estimator == "CME":
            pass
        elif cde_estimator in ["MDN", "LSCDE"]:
            treatment_type = active_dataset.dataset.treatment_type
            if treatment_type == "discrete":
                all_treatment = active_dataset.dataset.treatment
                all_adjustment = active_dataset.dataset.adjustment
                all_condition = active_dataset.dataset.condition
                all_adjustment = np.hstack([all_condition, all_adjustment])
                dim_adjustment = all_adjustment.shape[1]
                if cde is None:
                    if cde_estimator == "MDN":
                        cde = MixtureDensityNetwork(ndim_x=1, ndim_y=dim_adjustment, x_noise_std=0.0001, y_noise_std=0.0001, random_seed=None)
                    else:
                        cde = LSConditionalDensityEstimation(ndim_x=1, ndim_y=dim_adjustment,  bandwidth=0.05, random_seed=None)
                    cde.fit(all_treatment, all_adjustment)
                treatment_value = active_dataset.dataset.treatment_value
                treatment_value_array = np.array([treatment_value]*sampling_size).reshape(-1,1)
                sampled_adjustment = cde.sample(treatment_value_array)

                # Construct the target data
                unique_treatment = active_dataset.dataset.unique_treatment.reshape(-1,1)
                treatment_expanded = np.repeat(unique_treatment, sampling_size, axis=0)
                sampled_adjustment_expanded = np.tile(sampled_adjustment, (unique_treatment.shape[0], 1))
                condition_expanded = sampled_adjustment_expanded[:,:1]
                sampled_adjustment_expanded = sampled_adjustment_expanded[:,1:]
                target_data = {
                    "treatment": treatment_expanded,
                    "condition": condition_expanded,
                    "adjustment": sampled_adjustment_expanded,
                }
            elif treatment_type == "continuous":
                warnings.warn("The continuous treatment is not supported for the ATT task. The CME estimator will be used instead.", UserWarning)
            elif treatment_type == "binary":
                all_treatment = active_dataset.dataset.treatment
                all_adjustment = active_dataset.dataset.adjustment
                all_condition = active_dataset.dataset.condition
                all_adjustment = np.hstack([all_condition, all_adjustment])
                dim_adjustment = all_adjustment.shape[1]
                treatment_value = active_dataset.dataset.treatment_value
                # get the index of the treatment_value in all_treatment
                idx = np.where(all_treatment == treatment_value)[0][0]
                sampled_adjustment = all_adjustment[idx].reshape(1,-1)
                sampling_size = sampled_adjustment.shape[0]
                sampled_adjustment = sampled_adjustment
                
                # Construct the target data
                unique_treatment = active_dataset.dataset.unique_treatment.reshape(-1,1)
                treatment_expanded = np.repeat(unique_treatment, sampling_size, axis=0)
                sampled_adjustment_expanded = np.tile(sampled_adjustment, (unique_treatment.shape[0], 1))
                condition_expanded = sampled_adjustment_expanded[:,:1]
                sampled_adjustment_expanded = sampled_adjustment_expanded[:,1:]
                target_data = {
                    "treatment": treatment_expanded,
                    "condition": condition_expanded,
                    "adjustment": sampled_adjustment_expanded,
                }
            else:
                raise ValueError("The treatment_type should be either 'discrete', 'continuous' or 'binary'.")
    else:
        raise ValueError("The task_type should be either 'cate', 'ate' or 'att' or 'ds'.")
    
    # get the idx_list.
    if batch_aware == "B":
        var_list = []
        fast_component = {}
        for i in range(len(pool_idx_fix)):
            new_data = {
                        "treatment": active_dataset.dataset.treatment[pool_idx_fix[i]].reshape(1,-1),
                        "condition": active_dataset.dataset.condition[pool_idx_fix[i]].reshape(1,-1),
                        "adjustment": active_dataset.dataset.adjustment[pool_idx_fix[i]].reshape(1,-1),
                    }
                
            if cde_estimator == "CME":
                cov_i, fast_component = model.compute_variance(temp_data, new_data, fast_component, active_dataset)
            elif cde_estimator in ["MDN", "LSCDE"]:
                cov_i, fast_component = model.compute_cde_variance(temp_data, new_data, fast_component, target_data, sampling_size)
            else:
                raise ValueError("The cde_estimator should be either 'cme', 'mdn' or 'lscde'.")
            
            if adaptive_strategy == "VR":
                var_i = cov_i.trace()
            elif adaptive_strategy == "IG":
                sign, var_i = np.linalg.slogdet(cov_i)
                if sign <= 0:
                    raise ValueError("The determinant is non-positive, log-determinant is undefined.")
            else:
                raise ValueError("The adaptive_strategy should be either 'VR' or 'IG'.")
            
            var_list.append(var_i)
        
        idx_list = np.argsort(var_list)[:acq_size]
        
    elif batch_aware == "G":
        for _ in range(acq_size):
            var_list = []
            #NOTE: it would be good if we can use the fast_component from the previous iteration, easy to implement. I am lazy.
            fast_component = {} 
            for i in range(len(pool_idx_fix)):
                if i in idx_list:
                    var_list.append(1e10)
                    continue
                else:
                    new_data = {
                        "treatment": active_dataset.dataset.treatment[pool_idx_fix[i]].reshape(1,-1),
                        "condition": active_dataset.dataset.condition[pool_idx_fix[i]].reshape(1,-1),
                        "adjustment": active_dataset.dataset.adjustment[pool_idx_fix[i]].reshape(1,-1),
                    }

                    if cde_estimator == "CME":
                        cov_i, fast_component = model.compute_variance(temp_data, new_data, fast_component, active_dataset)
                    elif cde_estimator in ["MDN", "LSCDE"]:
                        cov_i, fast_component = model.compute_cde_variance(temp_data, new_data, fast_component, target_data, sampling_size)
                    else:
                        raise ValueError("The cde_estimator should be either 'cme', 'mdn' or 'lscde'.")
                    

                    if adaptive_strategy == "VR":
                        var_i = cov_i.trace()
                    elif adaptive_strategy == "IG":
                        sign, var_i = np.linalg.slogdet(cov_i)
                        if sign <= 0:
                            raise ValueError("The determinant is non-positive, log-determinant is undefined.")
                    else:
                        raise ValueError("The adaptive_strategy should be either 'VR' or 'IG'.")

                    var_list.append(var_i)
            
            # get the smallest index
            idx = np.argmin(var_list) # idx is the index of the pool_idx, not the index of the dataset
            temp_data = {
                "treatment": np.vstack([temp_data["treatment"], 
                                        active_dataset.dataset.treatment[pool_idx_fix[idx]]]),
                "condition": np.vstack([temp_data["condition"],
                                        active_dataset.dataset.condition[pool_idx_fix[idx]]]),
                "adjustment": np.vstack([temp_data["adjustment"],
                                        active_dataset.dataset.adjustment[pool_idx_fix[idx]]]),
            }
            idx_list.append(idx)
    else:
        raise ValueError("The batch_aware should be either 'B' or 'G'.")

    
    return idx_list, cde


def acqe(model, active_dataset, acq_size, condition_type, config=None, cde=None):
    """
    idx and idx_list are the index of the pool_idx, not the index of the dataset.
    Inside the active_dataset, the idx would be transformed to the index of the dataset.
    """

    # copy the active dataset
    active_dataset = copy.deepcopy(active_dataset)
    pool_idx_fix = active_dataset.pool_dataset.indices

    assert len(pool_idx_fix) >= acq_size, "The pool size is smaller than the acquisition size."

    adaptive_strategy = config.get("adaptive_strategy")
    cde_estimator = config.get("cde_estimator")
    batch_aware = config.get("batch_aware")

    ################################################################################################
    # Adaptive strategy
    ################################################################################################
    idx_list, cde = acqe_run(model, active_dataset, acq_size, cde_estimator, batch_aware, pool_idx_fix, cde, adaptive_strategy)

    #NOTE: Visualize the acquired data, Not important.
    if model.task_type in ["cate", "scate"]:
        # Analyze the acquired data
        required_idx_in_all_dataset = pool_idx_fix[idx_list]
        selected_condition = active_dataset.dataset.condition[required_idx_in_all_dataset]

        if condition_type == "continuous":
            # get the mean of the absolute value of the selected_condition
            mean_value = np.mean(selected_condition)
            variance_value = np.var(selected_condition)

            print("value mean:", mean_value)
            print("value variance:", variance_value)

        else:
            # Compute the number of each discrete value inside selected_condition
            unique_values, counts = np.unique(selected_condition, return_counts=True)
            # Print each value and its count
            for value, count in zip(unique_values, counts):
                print(f"Value: {value}, Count: {count}")
                
    return idx_list, cde

    
FUNCTIONS = {

    # New acquisition functions
    "random": random,
    "var_rank": var_rank,
    "coresets": coresets,
    "distance_rank": distance_rank,
    "var_reduction_rank": var_reduction_rank,
    "acqe": acqe,
}
