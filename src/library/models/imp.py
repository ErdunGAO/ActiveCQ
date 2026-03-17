import copy
from gpytorch.settings import fast_pred_samples
from numpy.core.numeric import ones
import torch
from torch import optim

from abc import ABC
import numpy as np

import gpytorch
from gpytorch import mlls
from src.library.modules import gaussian_process
from src.library.modules import CME
from tqdm import tqdm

from src.library.models.utils import expand_inverse_with_regularization
from src.library.modules.gaussian_process import DeltaKernel

import warnings

class IMP_CATE(ABC):
    def __init__(
        self,
        job_dir,
        task_type,
        condition_type,
        treatment_type,
        active_dataset,
        tune_dataset,
        learning_rate,
        epochs_gp,
        epochs_cme,
        learn_cme,
        patience,
        num_workers,
        device,
        kernel_type,
        seed,
    ):
        super(IMP_CATE, self).__init__()
        self.job_dir = job_dir
        self.task_type = task_type
        self.condition_type = condition_type
        self.treatment_type = treatment_type
        self.active_dataset = active_dataset
        self.tune_dataset = tune_dataset
        self.learning_rate = learning_rate
        self.best_loss = 1e7
        self.patience = patience
        self.counter = 0
        self.epochs_gp = epochs_gp
        self.epochs_cme = epochs_cme
        self.learn_cme = learn_cme
        self.num_workers = num_workers
        self.device = device
        self.kernel_type = kernel_type
        self.seed = seed
        #######################################################################
        # The inner GP model related stuff
        #NOTE: when you are using the gpytorch model, plase pay attention to the dimension of the input data.
        
        self.inputs = torch.tensor(active_dataset.extract_active_data()[0]).double()
        self.targets = torch.tensor(active_dataset.extract_active_data()[1]).double().squeeze()
        self.inputs_val = torch.tensor(tune_dataset.inputs).double()
        self.targets_val = torch.tensor(tune_dataset.targets).double().squeeze()

        self.likelihood_gp = gpytorch.likelihoods.GaussianLikelihood()
        self.model_gp = gaussian_process.ExactMultiInputGPModel_cate(
            train_x=self.inputs,
            train_y=self.targets,
            task_type=self.task_type,
            condition_type=self.condition_type,
            treatment_type=self.treatment_type,
            dim_treatment=self.active_dataset.dataset.dim_treatment,
            dim_condition=self.active_dataset.dataset.dim_condition,
            dim_adjustment=self.active_dataset.dataset.dim_adjustment,
            likelihood=self.likelihood_gp,
            kernel_type=self.kernel_type
        ).to(self.device)
        self.optimizer_gp = optim.Adam(self.model_gp.parameters(),
                                    lr=self.learning_rate)
        self.mll_gp = mlls.ExactMarginalLogLikelihood(self.likelihood_gp, self.model_gp)
        
        #######################################################################


        #######################################################################
        # The inner CME model related stuff
        # Data preparation
        self.treatment = torch.tensor(self.active_dataset.dataset.inputs_set["treatment"]).double()
        self.condition = torch.tensor(self.active_dataset.dataset.inputs_set["condition"]).double()
        self.adjustment = torch.tensor(self.active_dataset.dataset.inputs_set["adjustment"]).double()

        self.lambda_cme = torch.Tensor([np.sqrt(0.01)]).double()
        self.model_cme = CME.CME_learner(
            x=self.condition,
            y=self.adjustment,
            lambda_init=self.lambda_cme,
            device=self.device,
            kernel_x=self.kernel_type,
            kernel_y=self.kernel_type,
        ).to(self.device)

        self.optimizer_cme = optim.Adam(self.model_cme.parameters(),
                                    lr=self.learning_rate)
        self.loss_cme = CME.nll
        #######################################################################


    def fit_gp(self, exp_mode):

        for _ in range(self.epochs_gp):

            self.model_gp.train()
            self.likelihood_gp.train()

            inputs, targets = self.inputs.to(self.device), self.targets.to(self.device)
            self.optimizer_gp.zero_grad()
            output = self.model_gp(inputs)
            loss = -self.mll_gp(output, targets)
            loss.backward()
            self.optimizer_gp.step()

            # validation loss
            if _ % 10 == 0:
                self.model_gp.eval()
                self.likelihood_gp.eval()
                with torch.no_grad(), gpytorch.settings.fast_pred_var():

                    inputs_val, targets_val = self.inputs_val.to(self.device), self.targets_val.to(self.device)
                    output_val = self.likelihood_gp(self.model_gp(inputs_val))
                    loss_val = -self.mll_gp(output_val, targets_val)
                    # Early stopping after 300 epochs
                    if loss_val < self.best_loss:
                        self.best_loss = loss_val
                        self.counter = 0
                    else:
                        if _ > 200:
                            self.counter += 1
                    if self.counter > self.patience:
                        break
        
        self.model_gp.eval()
        self.likelihood_gp.eval()

        # if self.learn_cme and self.condition_type in ["continuous", "discrete"]:
        self.optimal_gp_condition_scale = self.model_gp.condition_kernel.lengthscale.detach()
        self.optimal_adjustment_scale = self.model_gp.adjustment_kernel.lengthscale.detach()

        if self.kernel_type == "rq":
            self.optimal_gp_condition_alpha = self.model_gp.condition_kernel.alpha.detach()
            self.optimal_adjustment_alpha = self.model_gp.adjustment_kernel.alpha.detach()

        self.optimal_inner_gp_noise = self.model_gp.likelihood.noise.detach()

    def fit_cme(self, exp_mode):

        self.model_cme.l_y.raw_lengthscale.requires_grad = False
        self.model_cme.l_y.lengthscale = self.optimal_adjustment_scale
        if self.kernel_type == "rq":
            self.model_cme.l_y.raw_alpha.requires_grad = False
            self.model_cme.l_y.alpha = self.optimal_adjustment_alpha

        self.model_cme.train()
        for _ in tqdm(range(self.epochs_cme)):
            self.optimizer_cme.zero_grad()
            train_K_lambda, train_L = self.model_cme()
            loss = self.loss_cme(train_K_lambda, train_L)
            loss.backward()
            self.optimizer_cme.step()
    
        self.model_cme.eval()
        self.optimal_lambda_cme = self.model_cme.lmda.detach()
        self.optimal_cme_condition_scale = self.model_cme.k_x.lengthscale.detach()
        if self.kernel_type == "rq":
            self.optimal_cme_condition_alpha = self.model_cme.k_x.alpha.detach()
        
    def save(self,):
        torch.save(self.model_gp.state_dict(), self.job_dir / "model_gp_best.pt")
        torch.save(self.model_cme.state_dict(), self.job_dir / "model_cme_best.pt")
    
    def load(self,):
        self.model_gp.load_state_dict(torch.load(self.job_dir / "model_gp_best.pt"))
        self.model_cme.load_state_dict(torch.load(self.job_dir / "model_cme_best.pt"))

    def fit(self, exp_mode):
        self.fit_gp(exp_mode=exp_mode)

        if self.learn_cme:
            self.fit_cme(exp_mode=exp_mode)
        self.save()

    #######################################################################
    # Baseline methods area starts here
    #######################################################################
    def init_fast_naive_component(self, temp_data, condition_value=None):
        fast_component = {}
        # condition_value_copy = torch.tensor(condition_value.detach().cpu().numpy()).double().reshape(-1, 1).to(self.device)
        # step 3.1: Compute the fast_X related term
        K_XX_t = self.k_x(self.treatment, temp_data["treatment"]).evaluate()
        X_related = {'K_XX_t': K_XX_t}
        fast_component['X_related'] = X_related

        # step 3.2: Compute the fast_Z related term
        K_ZZ_t = self.k_z(self.condition, temp_data["condition"]).evaluate()
        Z_related = {'K_ZZ_t': K_ZZ_t}
        fast_component['Z_related'] = Z_related

        # step 3.3: Compute the fast_S related term
        K_SS_t = self.k_s(self.adjustment, temp_data["adjustment"]).evaluate()
        S_related = {'K_SS_t': K_SS_t}
        fast_component['S_related'] = S_related

        w_t = K_XX_t * K_ZZ_t * K_SS_t
        w_related = {'w_t': w_t}
        fast_component['w_related'] = w_related

        # step 3.5: Compute the K_W_t related term
        K_X_t = self.k_x(temp_data["treatment"], temp_data["treatment"]).evaluate()
        K_Z_t = self.k_z(temp_data["condition"], temp_data["condition"]).evaluate()
        K_S_t = self.k_s(temp_data["adjustment"], temp_data["adjustment"]).evaluate()
        K_W_t = K_X_t * K_Z_t * K_S_t

        tilde_K_W_t = K_W_t + torch.eye(K_W_t.shape[0]).to(self.device) * self.optimal_inner_gp_noise
        inverse_tilde_K_W_t = torch.linalg.inv(tilde_K_W_t)
        W_related = {'K_W_t': K_W_t, 'tilde_K_W_t': tilde_K_W_t, 'inverse_tilde_K_W_t': inverse_tilde_K_W_t}
        fast_component['W_related'] = W_related

        return fast_component

    
    def compute_naive_variance(self, temp_data, new_data, fast_component, active_dataset):

        #Step 1: Prepare all the kernel objects and hyperparameters
        if self.kernel_type == "rbf":
            self.k_s = gpytorch.kernels.RBFKernel(
                lengthscale=self.optimal_adjustment_scale,
                ).to(self.device)
        elif self.kernel_type == "matern":
            self.k_s = gpytorch.kernels.MaternKernel(
                lengthscale=self.optimal_adjustment_scale,
                ).to(self.device)
        elif self.kernel_type == "rq":
            self.k_s = gpytorch.kernels.RQKernel(
                lengthscale=self.optimal_adjustment_scale,
                alpha=self.optimal_adjustment_alpha
                ).to(self.device)
        else:
            raise NotImplementedError("The kernel type is not supported.")
        
        #NOTE: we cannot directly use the condition kernel from the GP model since
        #the act_dim is different. We need to create a new kernel object.
        # k_z_gp = self.model_gp.condition_kernel
        if self.kernel_type == "rbf":
            self.k_z = gpytorch.kernels.RBFKernel(
                    lengthscale=self.optimal_gp_condition_scale,
                ).to(self.device)
        elif self.kernel_type == "matern":
            self.k_z = gpytorch.kernels.MaternKernel(
                    lengthscale=self.optimal_gp_condition_scale,
                ).to(self.device)
        elif self.kernel_type == "rq":
            self.k_z = gpytorch.kernels.RQKernel(
                    lengthscale=self.optimal_gp_condition_scale,
                    alpha=self.optimal_gp_condition_alpha
                ).to(self.device)
        else:
            raise NotImplementedError("The kernel type is not supported.")

        self.k_x = self.model_gp.treatment_kernel

        #Step 2: Prepare the data
        # step 2.1: Prepare the full data
        self.treatment = self.treatment.to(self.device)
        self.condition = self.condition.to(self.device)
        self.adjustment = self.adjustment.to(self.device)
        # step 2.2: Prepare the temp data
        temp_data = copy.deepcopy(temp_data)
        temp_data['treatment'] = torch.tensor(temp_data['treatment']).double().to(self.device)
        temp_data['condition'] = torch.tensor(temp_data['condition']).double().to(self.device)
        temp_data['adjustment'] = torch.tensor(temp_data['adjustment']).double().to(self.device)
        # step 2.3: Prepare the new data point
        new_data['treatment'] = torch.tensor(new_data['treatment']).double().to(self.device)
        new_data['condition'] = torch.tensor(new_data['condition']).double().to(self.device)
        new_data['adjustment'] = torch.tensor(new_data['adjustment']).double().to(self.device)

        #Step 3: Get the fast component term
        if len(fast_component) == 0: # if the fast component is empty
            fast_component = self.init_fast_naive_component(temp_data)
        
        #Step 4: Compute the variance
        k_Xx_t1 = self.k_x(self.treatment, new_data["treatment"]).evaluate()
        k_Zz_t1 = self.k_z(self.condition, new_data["condition"]).evaluate()
        k_Ss_t1 = self.k_s(self.adjustment, new_data["adjustment"]).evaluate()
        k_w_Xw_t1 = k_Xx_t1 * k_Zz_t1 * k_Ss_t1
        w_t1 = torch.concatenate([fast_component['w_related']['w_t'], k_w_Xw_t1], dim=1)

        K_X_tx_t1 = self.k_x(temp_data["treatment"], new_data["treatment"]).evaluate()
        K_Z_tz_t1 = self.k_z(temp_data["condition"], new_data["condition"]).evaluate()
        K_S_ts_t1 = self.k_s(temp_data["adjustment"], new_data["adjustment"]).evaluate()
        k_W_t1 = K_X_tx_t1 * K_Z_tz_t1 * K_S_ts_t1

        k_x_t1x_t1 = self.k_x(new_data["treatment"], new_data["treatment"]).evaluate()
        k_z_t1z_t1 = self.k_z(new_data["condition"], new_data["condition"]).evaluate()
        k_s_t1s_t1 = self.k_s(new_data["adjustment"], new_data["adjustment"]).evaluate()
        k_w_t1 = k_x_t1x_t1 * k_z_t1z_t1 * k_s_t1s_t1

        inverse_tilde_K_W_t1 = expand_inverse_with_regularization(
            fast_component['W_related']['inverse_tilde_K_W_t'],
            k_W_t1,
            k_w_t1,
            self.optimal_inner_gp_noise
        ).to(self.device)

        var = w_t1 @ inverse_tilde_K_W_t1 @ w_t1.t()

        return -var.trace().detach().cpu().numpy(), fast_component
    

    def compute_bald_variance(self, new_data):
        self.model_gp.eval()
        self.likelihood_gp.eval()
        self.model_cme.eval()

        new_data = torch.tensor(new_data).double().to(self.device)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            estimated_result = self.likelihood_gp(self.model_gp(new_data)).variance.detach()
            gamma_matrix = self.model_gp.covar_module(new_data).evaluate()
        
        return estimated_result.cpu().numpy(), gamma_matrix.cpu().numpy()

    #######################################################################
    # Baseline methods area ends here
    #######################################################################

    def init_fast_component(self, temp_data, condition_value=None):
        fast_component = {}
        # condition_value_copy = torch.tensor(condition_value.detach().cpu().numpy()).double().reshape(-1, 1).to(self.device)
        # step 3.1: Compute the fast_X related term
        K_XX_t = self.k_x(self.pseudo_treatment, temp_data["treatment"]).evaluate()
        X_related = {'K_XX_t': K_XX_t}
        fast_component['X_related'] = X_related
        # step 3.2: Compute the fast_Z related term
        k_zZ_t = self.k_z_gp(condition_value, temp_data["condition"]).evaluate()
        Z_related = {'k_zZ_t': k_zZ_t}
        fast_component['Z_related'] = Z_related
        # step 3.3: Compute the fast_S related term
        #NOTE: the above code sometimes will raise the error: AttributeError: 'LazyEvaluatedKernelTensor' object has no attribute 'add_diagonal'
        fixed_left = self.k_z_cme(condition_value, self.condition).evaluate() @ torch.linalg.inv(
            self.k_z_cme(self.condition, self.condition).evaluate() + torch.eye(self.condition.shape[0]).to(self.device) * self.optimal_lambda_cme)
        K_SS_t = self.k_s(self.adjustment, temp_data["adjustment"]).evaluate()
        S_related = {'fixed_left': fixed_left, 'K_SS_t': K_SS_t}
        fast_component['S_related'] = S_related
        # step 3.4: compute the w-related term
        w_t = K_XX_t * k_zZ_t * (fixed_left @ K_SS_t)
        w_related = {'w_t': w_t}

        fast_component['w_related'] = w_related
        # step 3.5: Compute the K_W_t related term
        K_X_t = self.k_x(temp_data["treatment"], temp_data["treatment"]).evaluate()
        K_Z_t = self.k_z_gp(temp_data["condition"], temp_data["condition"]).evaluate()
        K_S_t = self.k_s(temp_data["adjustment"], temp_data["adjustment"]).evaluate()
        K_W_t = K_X_t * K_Z_t * K_S_t
        tilde_K_W_t = K_W_t + torch.eye(K_W_t.shape[0]).to(self.device) * self.optimal_inner_gp_noise
        inverse_tilde_K_W_t = torch.linalg.inv(tilde_K_W_t)

        W_related = {'K_W_t': K_W_t, 'tilde_K_W_t': tilde_K_W_t, 'inverse_tilde_K_W_t': inverse_tilde_K_W_t}
        fast_component['W_related'] = W_related

        # to get the K_W_target
        K_X_target = self.k_x(self.pseudo_treatment, self.pseudo_treatment).evaluate()
        K_Z_target = self.k_z_gp(condition_value, condition_value).evaluate()
        K_SS = self.k_s(self.adjustment, self.adjustment).evaluate()
        K_S_target = fixed_left @ K_SS @ fixed_left.t()
        K_W_target = K_X_target * K_Z_target * K_S_target

        fast_component['K_W_target'] = K_W_target

        return fast_component


    def compute_variance(self, temp_data, new_data, fast_component, active_dataset):
        
        condition_value = active_dataset.dataset.condition_value
        assert condition_value is not None

        #Step 1: Prepare all the kernel objects and hyperparameters
        # Generate kernel object
        if self.learn_cme:
            self.k_z_cme = self.model_cme.k_x
            self.k_s = self.model_cme.l_y
        else:
            if self.kernel_type == "rbf":
                self.k_s = gpytorch.kernels.RBFKernel(
                    lengthscale=self.optimal_adjustment_scale,
                ).to(self.device)
            elif self.kernel_type == "matern":
                self.k_s = gpytorch.kernels.MaternKernel(
                    lengthscale=self.optimal_adjustment_scale,
                ).to(self.device)
            elif self.kernel_type == "rq":
                self.k_s = gpytorch.kernels.RQKernel(
                    lengthscale=self.optimal_adjustment_scale,
                    alpha=self.optimal_adjustment_alpha
                ).to(self.device)
            else:
                raise NotImplementedError("The kernel type is not supported.")
        
        #NOTE: we cannot directly use the condition kernel from the GP model since
        #the act_dim is different. We need to create a new kernel object.
        # k_z_gp = self.model_gp.condition_kernel
        if self.learn_cme and self.condition_type in ["continuous", "discrete"]:
            if self.kernel_type == "rbf":
                self.k_z_gp = gpytorch.kernels.RBFKernel(
                    lengthscale=self.optimal_gp_condition_scale,
                ).to(self.device)
            elif self.kernel_type == "matern":
                self.k_z_gp = gpytorch.kernels.MaternKernel(
                    lengthscale=self.optimal_gp_condition_scale,
                ).to(self.device)
            elif self.kernel_type == "rq":
                self.k_z_gp = gpytorch.kernels.RQKernel(
                    lengthscale=self.optimal_gp_condition_scale,
                    alpha=self.optimal_gp_condition_alpha
                ).to(self.device)
            else:
                raise NotImplementedError("The kernel type is not supported.")
        elif self.condition_type == "binary":
            self.k_z_gp = DeltaKernel()
        else:
            raise NotImplementedError("The condition type is not supported.")
        
        self.k_x = self.model_gp.treatment_kernel

        #Step 2: Prepare the data
        # step 2.1: Prepare the full data
        self.treatment = self.treatment.to(self.device)
        treatment_interest = active_dataset.dataset.treatment_interest
        # treatment_interest = torch.tensor(treatment_interest).double().to(self.device)
        if self.treatment_type == "binary":
            if treatment_interest is None:
                self.pseudo_treatment = torch.cat([torch.zeros(1), torch.ones(1)], dim=0).reshape(-1,1).to(self.device)
            else:
                self.pseudo_treatment = torch.tensor(treatment_interest).double().reshape(-1,1).to(self.device)
        else:
            if treatment_interest is None:
                self.pseudo_treatment = torch.unique(self.treatment).view(-1, 1)
            else:
                self.pseudo_treatment = torch.tensor(treatment_interest).double().reshape(-1,1).to(self.device)
        self.condition = self.condition.to(self.device)
        self.adjustment = self.adjustment.to(self.device)
        # step 2.2: Prepare the temp data
        temp_data = copy.deepcopy(temp_data)
        temp_data['treatment'] = torch.tensor(temp_data['treatment']).double().to(self.device)
        temp_data['condition'] = torch.tensor(temp_data['condition']).double().to(self.device)
        temp_data['adjustment'] = torch.tensor(temp_data['adjustment']).double().to(self.device)
        condition_value = torch.tensor(condition_value).double().reshape(1,1).to(self.device)
        # condition_value = torch.tensor(condition_value, dtype=torch.double, requires_grad=False).reshape(-1, 1).to(self.device).detach()
        # step 2.3: Prepare the new data point
        new_data['treatment'] = torch.tensor(new_data['treatment']).double().to(self.device)
        new_data['condition'] = torch.tensor(new_data['condition']).double().to(self.device)
        new_data['adjustment'] = torch.tensor(new_data['adjustment']).double().to(self.device)

        #Step 3: Get the fast component term
        if len(fast_component) == 0: # if the fast component is empty
            fast_component = self.init_fast_component(temp_data, condition_value)
        
        #Step 4: Compute the variance
        k_Xx_t1 = self.k_x(self.pseudo_treatment, new_data["treatment"]).evaluate()
        k_zz_t1 = self.k_z_gp(condition_value, new_data["condition"]).evaluate()
        s_part_s_t1 = fast_component['S_related']['fixed_left'] @ self.k_s(self.adjustment, new_data["adjustment"]).evaluate()
        k_w_Xw_t1 = k_Xx_t1 * k_zz_t1 * s_part_s_t1
        w_t1 = torch.concatenate([fast_component['w_related']['w_t'], k_w_Xw_t1], dim=1)


        K_X_tx_t1 = self.k_x(temp_data["treatment"], new_data["treatment"]).evaluate()
        K_Z_tz_t1 = self.k_z_gp(temp_data["condition"], new_data["condition"]).evaluate()
        K_S_ts_t1 = self.k_s(temp_data["adjustment"], new_data["adjustment"]).evaluate()
        k_W_t1 = K_X_tx_t1 * K_Z_tz_t1 * K_S_ts_t1

        k_x_t1x_t1 = self.k_x(new_data["treatment"], new_data["treatment"]).evaluate()
        k_z_t1z_t1 = self.k_z_gp(new_data["condition"], new_data["condition"]).evaluate()
        k_s_t1s_t1 = self.k_s(new_data["adjustment"], new_data["adjustment"]).evaluate()
        k_w_t1 = k_x_t1x_t1 * k_z_t1z_t1 * k_s_t1s_t1

        inverse_tilde_K_W_t1 = expand_inverse_with_regularization(
            fast_component['W_related']['inverse_tilde_K_W_t'],
            k_W_t1,
            k_w_t1,
            self.optimal_inner_gp_noise
        ).to(self.device)

        var = w_t1 @ inverse_tilde_K_W_t1 @ w_t1.t()

        epsilon = 1e-6
        K_W_target = fast_component['K_W_target']
        cov = K_W_target - var + epsilon * torch.eye(var.size(0), device=var.device)

        return cov.detach().cpu().numpy(), fast_component

        # return -var.trace().detach().cpu().numpy(), fast_component

    def init_fast_cde_component(self, temp_data, target_data):
        fast_component = {}
        # condition_value_copy = torch.tensor(condition_value.detach().cpu().numpy()).double().reshape(-1, 1).to(self.device)
        # step 3.1: Compute the fast_X related term
        K_XX_t = self.k_x(target_data["treatment"], temp_data["treatment"]).evaluate()
        X_related = {'K_XX_t': K_XX_t}
        fast_component['X_related'] = X_related

        # step 3.2: Compute the fast_Z related term
        K_ZZ_t = self.k_z(target_data["condition"], temp_data["condition"]).evaluate()
        Z_related = {'K_ZZ_t': K_ZZ_t}
        fast_component['Z_related'] = Z_related

        # step 3.3: Compute the fast_S related term
        K_SS_t = self.k_s(target_data["adjustment"], temp_data["adjustment"]).evaluate()
        S_related = {'K_SS_t': K_SS_t}
        fast_component['S_related'] = S_related

        w_t = K_XX_t * K_ZZ_t * K_SS_t
        w_related = {'w_t': w_t}
        fast_component['w_related'] = w_related

        # step 3.5: Compute the K_W_t related term
        K_X_t = self.k_x(temp_data["treatment"], temp_data["treatment"]).evaluate()
        K_Z_t = self.k_z(temp_data["condition"], temp_data["condition"]).evaluate()
        K_S_t = self.k_s(temp_data["adjustment"], temp_data["adjustment"]).evaluate()
        K_W_t = K_X_t * K_Z_t * K_S_t

        tilde_K_W_t = K_W_t + torch.eye(K_W_t.shape[0]).to(self.device) * self.optimal_inner_gp_noise
        inverse_tilde_K_W_t = torch.linalg.inv(tilde_K_W_t)

        W_related = {'K_W_t': K_W_t, 'tilde_K_W_t': tilde_K_W_t, 'inverse_tilde_K_W_t': inverse_tilde_K_W_t}
        fast_component['W_related'] = W_related
        
        K_X_target = self.k_x(target_data["treatment"], target_data["treatment"]).evaluate()
        K_Z_target = self.k_z(target_data["condition"], target_data["condition"]).evaluate()
        K_S_target = self.k_s(target_data["adjustment"], target_data["adjustment"]).evaluate()
        K_W_target = K_X_target * K_Z_target * K_S_target

        fast_component['K_W_target'] = K_W_target

        return fast_component
    

    def compute_cde_variance(self, temp_data, new_data, fast_component, target_data, sampling_size):

        #Step 1: Prepare all the kernel objects and hyperparameters
        if self.kernel_type == "rbf":
            self.k_s = gpytorch.kernels.RBFKernel(
                lengthscale=self.optimal_adjustment_scale,
                ).to(self.device)
        elif self.kernel_type == "matern":
            self.k_s = gpytorch.kernels.MaternKernel(
                lengthscale=self.optimal_adjustment_scale,
                ).to(self.device)
        elif self.kernel_type == "rq":
            self.k_s = gpytorch.kernels.RQKernel(
                lengthscale=self.optimal_adjustment_scale,
                alpha=self.optimal_adjustment_alpha
                ).to(self.device)
        else:
            raise NotImplementedError("The kernel type is not supported.")
        
        #NOTE: we cannot directly use the condition kernel from the GP model since
        #the act_dim is different. We need to create a new kernel object.
        # k_z_gp = self.model_gp.condition_kernel
        if self.kernel_type == "rbf":
            self.k_z = gpytorch.kernels.RBFKernel(
                    lengthscale=self.optimal_gp_condition_scale,
                ).to(self.device)
        elif self.kernel_type == "matern":
            self.k_z = gpytorch.kernels.MaternKernel(
                    lengthscale=self.optimal_gp_condition_scale,
                ).to(self.device)
        elif self.kernel_type == "rq":
            self.k_z = gpytorch.kernels.RQKernel(
                    lengthscale=self.optimal_gp_condition_scale,
                    alpha=self.optimal_gp_condition_alpha
                ).to(self.device)
        else:
            raise NotImplementedError("The kernel type is not supported.")

        self.k_x = self.model_gp.treatment_kernel

        #Step 2: Prepare the data
        # step 2.1: Prepare the full data
        target_data = copy.deepcopy(target_data)
        target_data['treatment'] = torch.tensor(target_data['treatment']).double().to(self.device)
        target_data['condition'] = torch.tensor(target_data['condition']).double().to(self.device)
        target_data['adjustment'] = torch.tensor(target_data['adjustment']).double().to(self.device)
        # step 2.2: Prepare the temp data
        temp_data = copy.deepcopy(temp_data)
        temp_data['treatment'] = torch.tensor(temp_data['treatment']).double().to(self.device)
        temp_data['condition'] = torch.tensor(temp_data['condition']).double().to(self.device)
        temp_data['adjustment'] = torch.tensor(temp_data['adjustment']).double().to(self.device)
        # step 2.3: Prepare the new data point
        new_data['treatment'] = torch.tensor(new_data['treatment']).double().to(self.device)
        new_data['condition'] = torch.tensor(new_data['condition']).double().to(self.device)
        new_data['adjustment'] = torch.tensor(new_data['adjustment']).double().to(self.device)

        #Step 3: Get the fast component term
        if len(fast_component) == 0: # if the fast component is empty
            fast_component = self.init_fast_cde_component(temp_data, target_data)
        
        #Step 4: Compute the variance
        k_Xx_t1 = self.k_x(target_data["treatment"], new_data["treatment"]).evaluate()
        k_Zz_t1 = self.k_z(target_data["condition"], new_data["condition"]).evaluate()
        k_Ss_t1 = self.k_s(target_data["adjustment"], new_data["adjustment"]).evaluate()
        k_w_Xw_t1 = k_Xx_t1 * k_Zz_t1 * k_Ss_t1
        w_t1 = torch.concatenate([fast_component['w_related']['w_t'], k_w_Xw_t1], dim=1)

        K_X_tx_t1 = self.k_x(temp_data["treatment"], new_data["treatment"]).evaluate()
        K_Z_tz_t1 = self.k_z(temp_data["condition"], new_data["condition"]).evaluate()
        K_S_ts_t1 = self.k_s(temp_data["adjustment"], new_data["adjustment"]).evaluate()
        k_W_t1 = K_X_tx_t1 * K_Z_tz_t1 * K_S_ts_t1

        k_x_t1x_t1 = self.k_x(new_data["treatment"], new_data["treatment"]).evaluate()
        k_z_t1z_t1 = self.k_z(new_data["condition"], new_data["condition"]).evaluate()
        k_s_t1s_t1 = self.k_s(new_data["adjustment"], new_data["adjustment"]).evaluate()
        k_w_t1 = k_x_t1x_t1 * k_z_t1z_t1 * k_s_t1s_t1

        inverse_tilde_K_W_t1 = expand_inverse_with_regularization(
            fast_component['W_related']['inverse_tilde_K_W_t'],
            k_W_t1,
            k_w_t1,
            self.optimal_inner_gp_noise
        ).to(self.device)

        var = w_t1 @ inverse_tilde_K_W_t1 @ w_t1.t()

        # return -var.trace().detach().cpu().numpy(), fast_component
        epsilon = 1e-6
        K_W_target = fast_component['K_W_target']
        cov = K_W_target - var + epsilon * torch.eye(var.size(0), device=var.device)

        cov = cov.view(cov.size(0) // sampling_size, sampling_size, -1, sampling_size).mean(dim=(1, 3))

        return cov.detach().cpu().numpy(), fast_component

        # return torch.logdet(var).detach().cpu().numpy(), fast_component
    

    def predict(self, ds_test):
        self.model_gp.eval()
        self.likelihood_gp.eval()
        self.model_cme.eval()

        # # For the continuous condition, it is not necessary to test on the training data.
        # if ds_test.condition_type == "continuous" and ds_test.dataset_type == "train":
        #     return 2024
        
        condition = ds_test.inputs_set["condition"]
        idx = np.where(condition == ds_test.condition_value)[0]

        condition = torch.tensor(condition[idx]).double().to(self.device)
        treatment = torch.tensor(ds_test.inputs_set["treatment"][idx]).double().to(self.device)
        adjustment = torch.tensor(ds_test.inputs_set["adjustment"][idx]).double().to(self.device)
        targets = torch.tensor(ds_test.targets[idx]).double().to(self.device)
        inputs = torch.tensor(ds_test.inputs[idx]).double().to(self.device)

        # Clustering the same treatment group
        treatment_unique = torch.unique(treatment).reshape(-1,1)

        amse_list = []
        for t in treatment_unique:
            idx_temp_t = torch.where(treatment == t)[0]
            # Predict the potential outcomes
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                estimated_result = self.likelihood_gp(self.model_gp(inputs[idx_temp_t])).mean.detach()
                # get the differnce between the potential outcomes
                se = (torch.mean(estimated_result) - torch.mean(targets[idx_temp_t]))**2
                amse_list.append(se)
        
        # return torch.mean(torch.tensor(amse_list)).cpu().numpy()
        return torch.sqrt(torch.mean(torch.tensor(amse_list))).cpu().numpy()

    




############################################################################################################
########################                 ATE           #####################################################
############################################################################################################
class IMP_DS(ABC):
    def __init__(
        self,
        job_dir,
        task_type,
        treatment_type,
        active_dataset,
        tune_dataset,
        learning_rate,
        epochs_gp,
        patience,
        num_workers,
        device,
        seed,
    ):
        super(IMP_DS, self).__init__()
        self.job_dir = job_dir
        self.task_type = task_type
        self.treatment_type = treatment_type
        self.active_dataset = active_dataset
        self.tune_dataset = tune_dataset
        self.learning_rate = learning_rate
        self.best_loss = 1e7
        self.patience = patience
        self.counter = 0
        self.epochs_gp = epochs_gp
        self.num_workers = num_workers
        self.device = device
        self.seed = seed

        #######################################################################
        # The inner GP model related stuff
        #NOTE: when you are using the gpytorch model, plase pay attention to the dimension of the input data.
        
        self.inputs = torch.tensor(active_dataset.extract_active_data()[0]).double()
        self.targets = torch.tensor(active_dataset.extract_active_data()[1]).double().squeeze()
        self.inputs_val = torch.tensor(tune_dataset.inputs).double()
        self.targets_val = torch.tensor(tune_dataset.targets).double().squeeze()

        if hasattr(active_dataset.dataset, 'dim_condition'):
            dim_adjustment = active_dataset.dataset.dim_adjustment + active_dataset.dataset.dim_condition
        else:
            dim_adjustment = active_dataset.dataset.dim_adjustment

        self.likelihood_gp = gpytorch.likelihoods.GaussianLikelihood()
        self.model_gp = gaussian_process.ExactMultiInputGPModel_ate_att(
            train_x=self.inputs,
            train_y=self.targets,
            task_type=self.task_type,
            treatment_type=self.treatment_type,
            dim_treatment=self.active_dataset.dataset.dim_treatment,
            dim_adjustment=dim_adjustment,
            likelihood=self.likelihood_gp,
        ).to(self.device)

        self.optimizer_gp = optim.Adam(self.model_gp.parameters(),
                                    lr=self.learning_rate)
        self.mll_gp = mlls.ExactMarginalLogLikelihood(self.likelihood_gp, self.model_gp)
        #######################################################################

        #######################################################################
        # Data preparation
        self.treatment = torch.tensor(self.active_dataset.dataset.inputs_set["treatment"]).double()
        self.condition = torch.tensor(self.active_dataset.dataset.inputs_set["condition"]).double()
        self.adjustment = torch.tensor(self.active_dataset.dataset.inputs_set["adjustment"]).double()
        self.adjustment = torch.cat([self.condition, self.adjustment], dim=1)
        self.ds_condition = torch.tensor(self.active_dataset.dataset.inputs_set["ds_condition"]).double()
        self.ds_adjustment = torch.tensor(self.active_dataset.dataset.inputs_set["ds_adjustment"]).double()
        self.ds_adjustment = torch.cat([self.ds_condition, self.ds_adjustment], dim=1)
        #######################################################################
    
    def preprocess(self, inputs, targets):
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        return inputs, targets
    

    def fit_gp(self, exp_mode):

        for _ in range(self.epochs_gp):

            self.model_gp.train()
            self.likelihood_gp.train()

            inputs, targets = self.inputs.to(self.device), self.targets.to(self.device)
            self.optimizer_gp.zero_grad()
            output = self.model_gp(inputs)
            loss = -self.mll_gp(output, targets)
            loss.backward()
            self.optimizer_gp.step()

            # validation loss
            if _ % 10 == 0:
                self.model_gp.eval()
                self.likelihood_gp.eval()
                with torch.no_grad(), gpytorch.settings.fast_pred_var():

                    inputs_val, targets_val = self.inputs_val.to(self.device), self.targets_val.to(self.device)
                    output_val = self.likelihood_gp(self.model_gp(inputs_val))
                    loss_val = -self.mll_gp(output_val, targets_val)
                    # Early stopping after 300 epochs
                    if loss_val < self.best_loss:
                        self.best_loss = loss_val
                        self.counter = 0
                    else:
                        if _ > 200:
                            self.counter += 1
                    if self.counter > self.patience:
                        break
        
        self.model_gp.eval()
        self.likelihood_gp.eval()

        self.optimal_adjustment_scale = self.model_gp.adjustment_kernel.lengthscale.detach()
        self.optimal_inner_gp_noise = self.model_gp.likelihood.noise.detach()
        
    def save(self,):
        torch.save(self.model_gp.state_dict(), self.job_dir / "model_gp_best.pt")
    
    def load(self,):
        self.model_gp.load_state_dict(torch.load(self.job_dir / "model_gp_best.pt"))

    def fit(self, exp_mode):
        self.fit_gp(exp_mode=exp_mode)
        self.save()

    #######################################################################
    # Baseline methods area starts here
    #######################################################################
    def init_fast_naive_component(self, temp_data, condition_value=None):
        fast_component = {}
        K_XX_t = self.k_x(self.treatment, temp_data["treatment"]).evaluate()
        X_related = {'K_XX_t': K_XX_t}
        fast_component['X_related'] = X_related

        # step 3.3: Compute the fast_S related term
        K_SS_t = self.k_s(self.adjustment, temp_data["adjustment"]).evaluate()
        S_related = {'K_SS_t': K_SS_t}
        fast_component['S_related'] = S_related

        w_t = K_XX_t * K_SS_t
        w_related = {'w_t': w_t}
        fast_component['w_related'] = w_related

        # step 3.5: Compute the K_W_t related term
        K_X_t = self.k_x(temp_data["treatment"], temp_data["treatment"]).evaluate()
        K_S_t = self.k_s(temp_data["adjustment"], temp_data["adjustment"]).evaluate()
        K_W_t = K_X_t * K_S_t

        tilde_K_W_t = K_W_t + torch.eye(K_W_t.shape[0]).to(self.device) * self.optimal_inner_gp_noise
        inverse_tilde_K_W_t = torch.linalg.inv(tilde_K_W_t)
        W_related = {'K_W_t': K_W_t, 'tilde_K_W_t': tilde_K_W_t, 'inverse_tilde_K_W_t': inverse_tilde_K_W_t}
        fast_component['W_related'] = W_related

        return fast_component

    
    def compute_naive_variance(self, temp_data, new_data, fast_component, active_dataset):

        #Step 1: Prepare all the kernel objects and hyperparameters
        self.k_s = gpytorch.kernels.RBFKernel(
            lengthscale=self.optimal_adjustment_scale,
            ).to(self.device)

        self.k_x = self.model_gp.treatment_kernel

        #Step 2: Prepare the data
        # step 2.1: Prepare the full data
        self.treatment = self.treatment.to(self.device)
        self.adjustment = self.adjustment.to(self.device)
        # step 2.2: Prepare the temp data
        temp_data = copy.deepcopy(temp_data)
        temp_data['treatment'] = torch.tensor(temp_data['treatment']).double().to(self.device)
        temp_data['condition'] = torch.tensor(temp_data['condition']).double().to(self.device)
        temp_data['adjustment'] = torch.tensor(temp_data['adjustment']).double().to(self.device)
        temp_data['adjustment'] = torch.cat([temp_data['condition'], temp_data['adjustment']], dim=1)
        # step 2.3: Prepare the new data point
        new_data['treatment'] = torch.tensor(new_data['treatment']).double().to(self.device)
        new_data['condition'] = torch.tensor(new_data['condition']).double().to(self.device)
        new_data['adjustment'] = torch.tensor(new_data['adjustment']).double().to(self.device)
        new_data['adjustment'] = torch.cat([new_data['condition'],new_data['adjustment']], dim=1)

        #Step 3: Get the fast component term
        if len(fast_component) == 0: # if the fast component is empty
            fast_component = self.init_fast_naive_component(temp_data)
        
        #Step 4: Compute the variance
        k_Xx_t1 = self.k_x(self.treatment, new_data["treatment"]).evaluate()
        k_Ss_t1 = self.k_s(self.adjustment, new_data["adjustment"]).evaluate()
        k_w_Xw_t1 = k_Xx_t1 * k_Ss_t1
        w_t1 = torch.concatenate([fast_component['w_related']['w_t'], k_w_Xw_t1], dim=1)

        K_X_tx_t1 = self.k_x(temp_data["treatment"], new_data["treatment"]).evaluate()
        K_S_ts_t1 = self.k_s(temp_data["adjustment"], new_data["adjustment"]).evaluate()
        k_W_t1 = K_X_tx_t1 * K_S_ts_t1

        k_x_t1x_t1 = self.k_x(new_data["treatment"], new_data["treatment"]).evaluate()
        k_s_t1s_t1 = self.k_s(new_data["adjustment"], new_data["adjustment"]).evaluate()
        k_w_t1 = k_x_t1x_t1 * k_s_t1s_t1

        inverse_tilde_K_W_t1 = expand_inverse_with_regularization(
            fast_component['W_related']['inverse_tilde_K_W_t'],
            k_W_t1,
            k_w_t1,
            self.optimal_inner_gp_noise
        ).to(self.device)

        var = w_t1 @ inverse_tilde_K_W_t1 @ w_t1.t()

        return -var.trace().detach().cpu().numpy(), fast_component
    
    def compute_bald_variance(self, new_data):
        self.model_gp.eval()
        self.likelihood_gp.eval()

        new_data = torch.tensor(new_data).double().to(self.device)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            estimated_result = self.likelihood_gp(self.model_gp(new_data)).variance.detach()
            gamma_matrix = self.model_gp.covar_module(new_data).evaluate()
        
        return estimated_result.cpu().numpy(), gamma_matrix.cpu().numpy()
    #######################################################################
    # Baseline methods area ends here
    #######################################################################








    def init_fast_component(self, temp_data, condition_value=None):
        fast_component = {}
        # step 3.1: Compute the fast_X related term
        K_XX_t = self.k_x(self.pseudo_treatment, temp_data["treatment"]).evaluate()
        X_related = {'K_XX_t': K_XX_t}
        fast_component['X_related'] = X_related
        # step 3.2: Compute the fast_Z related term
        # step 3.3: Compute the fast_S related term
        K_SS_t = self.k_s(self.ds_adjustment, temp_data["adjustment"]).evaluate()
        S_related = {'K_SS_t': K_SS_t}
        fast_component['S_related'] = S_related
        # step 3.4: compute the w-related term
        # step 3.5: Compute the K_W_t related term
        K_X_t = self.k_x(temp_data["treatment"], temp_data["treatment"]).evaluate()
        K_S_t = self.k_s(temp_data["adjustment"], temp_data["adjustment"]).evaluate()
        K_W_t = K_X_t * K_S_t
        tilde_K_W_t = K_W_t + torch.eye(K_W_t.shape[0]).to(self.device) * self.optimal_inner_gp_noise
        inverse_tilde_K_W_t = torch.linalg.inv(tilde_K_W_t)
        W_related = {'K_W_t': K_W_t, 'tilde_K_W_t': tilde_K_W_t, 'inverse_tilde_K_W_t': inverse_tilde_K_W_t}
        fast_component['W_related'] = W_related

        # to get the K_W_target
        K_X_target = self.k_x(self.pseudo_treatment, self.pseudo_treatment).evaluate()
        K_SS_target = self.k_s(self.ds_adjustment, self.ds_adjustment).evaluate()
        K_W_target = K_X_target * K_SS_target.mean()
        fast_component['K_W_target'] = K_W_target

        return fast_component


    def compute_variance(self, temp_data, new_data, fast_component, active_dataset):

        #Step 1: Prepare all the kernel objects and hyperparameters
        # Generate kernel object

        #NOTE: we cannot directly use the condition kernel from the GP model since
        #the act_dim is different. We need to create a new kernel object.
        # k_z_gp = self.model_gp.condition_kernel
        self.k_s = gpytorch.kernels.RBFKernel(
            lengthscale=self.optimal_adjustment_scale,
        ).to(self.device)
        self.k_x = self.model_gp.treatment_kernel

        #Step 2: Prepare the data
        # step 2.1: Prepare the full data
        self.treatment = self.treatment.to(self.device)
        treatment_interest = active_dataset.dataset.treatment_interest
        # treatment_interest = torch.tensor(treatment_interest).double().to(self.device)
        if self.treatment_type == "binary":
            if treatment_interest is None:
                self.pseudo_treatment = torch.cat([torch.zeros(1), torch.ones(1)], dim=0).reshape(-1,1).to(self.device)
            else:
                self.pseudo_treatment = torch.tensor(treatment_interest).double().reshape(-1,1).to(self.device)
        elif self.treatment_type == "discrete":
            if treatment_interest is None:
                self.pseudo_treatment = torch.unique(self.treatment).view(-1, 1)
            else:
                self.pseudo_treatment = torch.tensor(treatment_interest).double().reshape(-1,1).to(self.device)
        elif self.treatment_type == "continuous":
            assert treatment_interest is None
            self.pseudo_treatment = self.treatment

        self.adjustment = self.adjustment.to(self.device) #useless in this case
        self.ds_adjustment = self.ds_adjustment.to(self.device)
        # step 2.2: Prepare the temp data
        temp_data = copy.deepcopy(temp_data)
        temp_data['treatment'] = torch.tensor(temp_data['treatment']).double().to(self.device)
        temp_data['condition'] = torch.tensor(temp_data['condition']).double().to(self.device)
        temp_data['adjustment'] = torch.tensor(temp_data['adjustment']).double().to(self.device)
        temp_data['adjustment'] = torch.cat([temp_data['condition'], temp_data['adjustment']], dim=1)
        # step 2.3: Prepare the new data point
        new_data['treatment'] = torch.tensor(new_data['treatment']).double().to(self.device)
        new_data['condition'] = torch.tensor(new_data['condition']).double().to(self.device)
        new_data['adjustment'] = torch.tensor(new_data['adjustment']).double().to(self.device)
        new_data['adjustment'] = torch.cat([new_data['condition'],new_data['adjustment']], dim=1)

        #Step 3: Get the fast component term
        if len(fast_component) == 0: # if the fast component is empty
            fast_component = self.init_fast_component(temp_data)
        
        #Step 4: Compute the variance
        k_Xx_t1 = self.k_x(self.pseudo_treatment, new_data["treatment"]).evaluate()
        k_Ss_t1 = self.k_s(self.ds_adjustment, new_data["adjustment"]).evaluate()
        K_XX_t1 = torch.concatenate([fast_component['X_related']['K_XX_t'], k_Xx_t1], dim=1)
        K_SS_t1 = torch.concatenate([fast_component['S_related']['K_SS_t'], k_Ss_t1], dim=1)
        n = self.treatment.shape[0]
        w_t1 = K_XX_t1 * (torch.ones(1, n, device=self.device, dtype=torch.double) / n @ K_SS_t1)

        K_X_tx_t1 = self.k_x(temp_data["treatment"], new_data["treatment"]).evaluate()
        K_S_ts_t1 = self.k_s(temp_data["adjustment"], new_data["adjustment"]).evaluate()
        k_W_t1 = K_X_tx_t1 * K_S_ts_t1

        k_x_t1x_t1 = self.k_x(new_data["treatment"], new_data["treatment"]).evaluate()
        k_s_t1s_t1 = self.k_s(new_data["adjustment"], new_data["adjustment"]).evaluate()
        k_w_t1 = k_x_t1x_t1 * k_s_t1s_t1

        inverse_tilde_K_W_t1 = expand_inverse_with_regularization(
            fast_component['W_related']['inverse_tilde_K_W_t'],
            k_W_t1,
            k_w_t1,
            self.optimal_inner_gp_noise
        ).to(self.device)

        var = w_t1 @ inverse_tilde_K_W_t1 @ w_t1.t()

        epsilon = 1e-6
        K_W_target = fast_component['K_W_target']
        cov = K_W_target - var + epsilon * torch.eye(var.size(0), device=var.device)

        return cov.detach().cpu().numpy(), fast_component
        # return -var.trace().detach().cpu().numpy(), fast_component


    def init_fast_cde_component(self, temp_data, target_data):
        fast_component = {}
        # condition_value_copy = torch.tensor(condition_value.detach().cpu().numpy()).double().reshape(-1, 1).to(self.device)
        # step 3.1: Compute the fast_X related term
        K_XX_t = self.k_x(target_data["treatment"], temp_data["treatment"]).evaluate()
        X_related = {'K_XX_t': K_XX_t}
        fast_component['X_related'] = X_related

        # step 3.3: Compute the fast_S related term
        K_SS_t = self.k_s(target_data["adjustment"], temp_data["adjustment"]).evaluate()
        S_related = {'K_SS_t': K_SS_t}
        fast_component['S_related'] = S_related

        w_t = K_XX_t * K_SS_t
        w_related = {'w_t': w_t}
        fast_component['w_related'] = w_related

        # step 3.5: Compute the K_W_t related term
        K_X_t = self.k_x(temp_data["treatment"], temp_data["treatment"]).evaluate()
        K_S_t = self.k_s(temp_data["adjustment"], temp_data["adjustment"]).evaluate()
        K_W_t = K_X_t * K_S_t

        tilde_K_W_t = K_W_t + torch.eye(K_W_t.shape[0]).to(self.device) * self.optimal_inner_gp_noise
        inverse_tilde_K_W_t = torch.linalg.inv(tilde_K_W_t)
        W_related = {'K_W_t': K_W_t, 'tilde_K_W_t': tilde_K_W_t, 'inverse_tilde_K_W_t': inverse_tilde_K_W_t}
        fast_component['W_related'] = W_related

        K_X_target = self.k_x(target_data["treatment"], target_data["treatment"]).evaluate()
        K_S_target = self.k_s(target_data["adjustment"], target_data["adjustment"]).evaluate()
        K_W_target = K_X_target * K_S_target

        fast_component['K_W_target'] = K_W_target

        return fast_component

    
    def compute_cde_variance(self, temp_data, new_data, fast_component, target_data, sampling_size):

        #Step 1: Prepare all the kernel objects and hyperparameters
        self.k_s = gpytorch.kernels.RBFKernel(
            lengthscale=self.optimal_adjustment_scale,
            ).to(self.device)

        self.k_x = self.model_gp.treatment_kernel

        #Step 2: Prepare the data
        # step 2.1: Prepare the full data
        target_data = copy.deepcopy(target_data)
        target_data['treatment'] = torch.tensor(target_data['treatment']).double().to(self.device)
        target_data['condition'] = torch.tensor(target_data['condition']).double().to(self.device)
        target_data['adjustment'] = torch.tensor(target_data['adjustment']).double().to(self.device)
        target_data['adjustment'] = torch.cat([target_data['condition'], target_data['adjustment']], dim=1)
        # step 2.2: Prepare the temp data
        temp_data = copy.deepcopy(temp_data)
        temp_data['treatment'] = torch.tensor(temp_data['treatment']).double().to(self.device)
        temp_data['condition'] = torch.tensor(temp_data['condition']).double().to(self.device)
        temp_data['adjustment'] = torch.tensor(temp_data['adjustment']).double().to(self.device)
        temp_data['adjustment'] = torch.cat([temp_data['condition'], temp_data['adjustment']], dim=1)
        # step 2.3: Prepare the new data point
        new_data['treatment'] = torch.tensor(new_data['treatment']).double().to(self.device)
        new_data['condition'] = torch.tensor(new_data['condition']).double().to(self.device)
        new_data['adjustment'] = torch.tensor(new_data['adjustment']).double().to(self.device)
        new_data['adjustment'] = torch.cat([new_data['condition'],new_data['adjustment']], dim=1)

        #Step 3: Get the fast component term
        if len(fast_component) == 0: # if the fast component is empty
            fast_component = self.init_fast_cde_component(temp_data, target_data)
        
        #Step 4: Compute the variance
        k_Xx_t1 = self.k_x(target_data["treatment"], new_data["treatment"]).evaluate()
        k_Ss_t1 = self.k_s(target_data["adjustment"], new_data["adjustment"]).evaluate()
        k_w_Xw_t1 = k_Xx_t1 * k_Ss_t1
        w_t1 = torch.concatenate([fast_component['w_related']['w_t'], k_w_Xw_t1], dim=1)

        K_X_tx_t1 = self.k_x(temp_data["treatment"], new_data["treatment"]).evaluate()
        K_S_ts_t1 = self.k_s(temp_data["adjustment"], new_data["adjustment"]).evaluate()
        k_W_t1 = K_X_tx_t1 * K_S_ts_t1

        k_x_t1x_t1 = self.k_x(new_data["treatment"], new_data["treatment"]).evaluate()
        k_s_t1s_t1 = self.k_s(new_data["adjustment"], new_data["adjustment"]).evaluate()
        k_w_t1 = k_x_t1x_t1 * k_s_t1s_t1

        inverse_tilde_K_W_t1 = expand_inverse_with_regularization(
            fast_component['W_related']['inverse_tilde_K_W_t'],
            k_W_t1,
            k_w_t1,
            self.optimal_inner_gp_noise
        ).to(self.device)

        var = w_t1 @ inverse_tilde_K_W_t1 @ w_t1.t()

        # return -var.trace().detach().cpu().numpy(), fast_component
        epsilon = 1e-6
        K_W_target = fast_component['K_W_target']
        cov = K_W_target - var + epsilon * torch.eye(var.shape[0]).to(self.device)

        cov = cov.view(cov.size(0) // sampling_size, sampling_size, -1, sampling_size).mean(dim=(1, 3))

        return cov.detach().cpu().numpy(), fast_component

        # return torch.logdet(var).detach().cpu().numpy(), fast_component
    


    def predict(self, ds_test):
        self.model_gp.eval()
        self.likelihood_gp.eval()

        treatment = torch.tensor(ds_test.inputs_set["ds_treatment"]).double().to(self.device)
        targets = torch.tensor(ds_test.ds_targets).double().to(self.device)
        inputs = torch.tensor(ds_test.ds_inputs).double().to(self.device)

        if hasattr(ds_test, 'treatment_interest'):
            treatment_interest = ds_test.treatment_interest
        else:
            treatment_interest = ds_test.dataset.treatment_interest

        if treatment_interest is not None:
            mask = (treatment == treatment_interest).squeeze()
            treatment = treatment[mask]
            targets = targets[mask]
            inputs = inputs[mask]

        # Clustering the same treatment group
        treatment_unique = torch.unique(treatment).reshape(-1,1)

        amse_list = []
        for t in treatment_unique:
            idx_temp_t = torch.where(treatment == t)[0]
            # Predict the potential outcomes
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                estimated_result = self.likelihood_gp(self.model_gp(inputs[idx_temp_t])).mean.detach()
                # get the differnce between the potential outcomes
                se = (torch.mean(estimated_result) - torch.mean(targets[idx_temp_t]))**2
                amse_list.append(se)
        
        # return torch.mean(torch.tensor(amse_list)).cpu().numpy()
        return torch.sqrt(torch.mean(torch.tensor(amse_list))).cpu().numpy()







############################################################################################################
########################                 ATE           #####################################################
############################################################################################################
class IMP_ATE(ABC):
    def __init__(
        self,
        job_dir,
        task_type,
        treatment_type,
        active_dataset,
        tune_dataset,
        learning_rate,
        epochs_gp,
        patience,
        num_workers,
        device,
        seed,
    ):
        super(IMP_ATE, self).__init__()
        self.job_dir = job_dir
        self.task_type = task_type
        self.treatment_type = treatment_type
        self.active_dataset = active_dataset
        self.tune_dataset = tune_dataset
        self.learning_rate = learning_rate
        self.best_loss = 1e7
        self.patience = patience
        self.counter = 0
        self.epochs_gp = epochs_gp
        self.num_workers = num_workers
        self.device = device
        self.seed = seed

        #######################################################################
        # The inner GP model related stuff
        #NOTE: when you are using the gpytorch model, plase pay attention to the dimension of the input data.
        
        self.inputs = torch.tensor(active_dataset.extract_active_data()[0]).double()
        self.targets = torch.tensor(active_dataset.extract_active_data()[1]).double().squeeze()
        self.inputs_val = torch.tensor(tune_dataset.inputs).double()
        self.targets_val = torch.tensor(tune_dataset.targets).double().squeeze()

        if hasattr(active_dataset.dataset, 'dim_condition'):
            dim_adjustment = active_dataset.dataset.dim_adjustment + active_dataset.dataset.dim_condition
        else:
            dim_adjustment = active_dataset.dataset.dim_adjustment

        self.likelihood_gp = gpytorch.likelihoods.GaussianLikelihood()
        self.model_gp = gaussian_process.ExactMultiInputGPModel_ate_att(
            train_x=self.inputs,
            train_y=self.targets,
            task_type=self.task_type,
            treatment_type=self.treatment_type,
            dim_treatment=self.active_dataset.dataset.dim_treatment,
            dim_adjustment=dim_adjustment,
            likelihood=self.likelihood_gp,
        ).to(self.device)

        self.optimizer_gp = optim.Adam(self.model_gp.parameters(),
                                    lr=self.learning_rate)
        self.mll_gp = mlls.ExactMarginalLogLikelihood(self.likelihood_gp, self.model_gp)
        #######################################################################

        #######################################################################
        # Data preparation
        self.treatment = torch.tensor(self.active_dataset.dataset.inputs_set["treatment"]).double()
        self.condition = torch.tensor(self.active_dataset.dataset.inputs_set["condition"]).double()
        self.adjustment = torch.tensor(self.active_dataset.dataset.inputs_set["adjustment"]).double()
        self.adjustment = torch.cat([self.condition, self.adjustment], dim=1)
        #######################################################################
    
    def preprocess(self, inputs, targets):
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        return inputs, targets
    

    def fit_gp(self, exp_mode):

        for _ in range(self.epochs_gp):

            self.model_gp.train()
            self.likelihood_gp.train()

            inputs, targets = self.inputs.to(self.device), self.targets.to(self.device)
            self.optimizer_gp.zero_grad()
            output = self.model_gp(inputs)
            loss = -self.mll_gp(output, targets)
            loss.backward()
            self.optimizer_gp.step()

            # validation loss
            if _ % 10 == 0:
                self.model_gp.eval()
                self.likelihood_gp.eval()
                with torch.no_grad(), gpytorch.settings.fast_pred_var():

                    inputs_val, targets_val = self.inputs_val.to(self.device), self.targets_val.to(self.device)
                    output_val = self.likelihood_gp(self.model_gp(inputs_val))
                    loss_val = -self.mll_gp(output_val, targets_val)
                    # Early stopping after 300 epochs
                    if loss_val < self.best_loss:
                        self.best_loss = loss_val
                        self.counter = 0
                    else:
                        if _ > 200:
                            self.counter += 1
                    if self.counter > self.patience:
                        break
        
        self.model_gp.eval()
        self.likelihood_gp.eval()

        self.optimal_adjustment_scale = self.model_gp.adjustment_kernel.lengthscale.detach()
        self.optimal_inner_gp_noise = self.model_gp.likelihood.noise.detach()
        
    def save(self,):
        torch.save(self.model_gp.state_dict(), self.job_dir / "model_gp_best.pt")
    
    def load(self,):
        self.model_gp.load_state_dict(torch.load(self.job_dir / "model_gp_best.pt"))

    def fit(self, exp_mode):
        self.fit_gp(exp_mode=exp_mode)
        self.save()

    #######################################################################
    # Baseline methods area starts here
    #######################################################################
    def init_fast_naive_component(self, temp_data, condition_value=None):
        fast_component = {}
        K_XX_t = self.k_x(self.treatment, temp_data["treatment"]).evaluate()
        X_related = {'K_XX_t': K_XX_t}
        fast_component['X_related'] = X_related

        K_SS_t = self.k_s(self.adjustment, temp_data["adjustment"]).evaluate()
        S_related = {'K_SS_t': K_SS_t}
        fast_component['S_related'] = S_related

        w_t = K_XX_t * K_SS_t
        w_related = {'w_t': w_t}
        fast_component['w_related'] = w_related

        # step 3.5: Compute the K_W_t related term
        K_X_t = self.k_x(temp_data["treatment"], temp_data["treatment"]).evaluate()
        K_S_t = self.k_s(temp_data["adjustment"], temp_data["adjustment"]).evaluate()
        K_W_t = K_X_t * K_S_t

        tilde_K_W_t = K_W_t + torch.eye(K_W_t.shape[0]).to(self.device) * self.optimal_inner_gp_noise
        inverse_tilde_K_W_t = torch.linalg.inv(tilde_K_W_t)
        W_related = {'K_W_t': K_W_t, 'tilde_K_W_t': tilde_K_W_t, 'inverse_tilde_K_W_t': inverse_tilde_K_W_t}
        fast_component['W_related'] = W_related

        return fast_component

    
    def compute_naive_variance(self, temp_data, new_data, fast_component, active_dataset):

        #Step 1: Prepare all the kernel objects and hyperparameters
        self.k_s = gpytorch.kernels.RBFKernel(
            lengthscale=self.optimal_adjustment_scale,
            ).to(self.device)

        self.k_x = self.model_gp.treatment_kernel

        #Step 2: Prepare the data
        # step 2.1: Prepare the full data
        self.treatment = self.treatment.to(self.device)
        self.adjustment = self.adjustment.to(self.device)
        # step 2.2: Prepare the temp data
        temp_data = copy.deepcopy(temp_data)
        temp_data['treatment'] = torch.tensor(temp_data['treatment']).double().to(self.device)
        temp_data['condition'] = torch.tensor(temp_data['condition']).double().to(self.device)
        temp_data['adjustment'] = torch.tensor(temp_data['adjustment']).double().to(self.device)
        temp_data['adjustment'] = torch.cat([temp_data['condition'], temp_data['adjustment']], dim=1)
        # step 2.3: Prepare the new data point
        new_data['treatment'] = torch.tensor(new_data['treatment']).double().to(self.device)
        new_data['condition'] = torch.tensor(new_data['condition']).double().to(self.device)
        new_data['adjustment'] = torch.tensor(new_data['adjustment']).double().to(self.device)
        new_data['adjustment'] = torch.cat([new_data['condition'],new_data['adjustment']], dim=1)

        #Step 3: Get the fast component term
        if len(fast_component) == 0: # if the fast component is empty
            fast_component = self.init_fast_naive_component(temp_data)
        
        #Step 4: Compute the variance
        k_Xx_t1 = self.k_x(self.treatment, new_data["treatment"]).evaluate()
        k_Ss_t1 = self.k_s(self.adjustment, new_data["adjustment"]).evaluate()
        k_w_Xw_t1 = k_Xx_t1 * k_Ss_t1
        w_t1 = torch.concatenate([fast_component['w_related']['w_t'], k_w_Xw_t1], dim=1)

        K_X_tx_t1 = self.k_x(temp_data["treatment"], new_data["treatment"]).evaluate()
        K_S_ts_t1 = self.k_s(temp_data["adjustment"], new_data["adjustment"]).evaluate()
        k_W_t1 = K_X_tx_t1 * K_S_ts_t1

        k_x_t1x_t1 = self.k_x(new_data["treatment"], new_data["treatment"]).evaluate()
        k_s_t1s_t1 = self.k_s(new_data["adjustment"], new_data["adjustment"]).evaluate()
        k_w_t1 = k_x_t1x_t1 * k_s_t1s_t1

        inverse_tilde_K_W_t1 = expand_inverse_with_regularization(
            fast_component['W_related']['inverse_tilde_K_W_t'],
            k_W_t1,
            k_w_t1,
            self.optimal_inner_gp_noise
        ).to(self.device)

        var = w_t1 @ inverse_tilde_K_W_t1 @ w_t1.t()

        return -var.trace().detach().cpu().numpy(), fast_component


    def compute_bald_variance(self, new_data):
        self.model_gp.eval()
        self.likelihood_gp.eval()

        new_data = torch.tensor(new_data).double().to(self.device)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            estimated_result = self.likelihood_gp(self.model_gp(new_data)).variance.detach()
            gamma_matrix = self.model_gp.covar_module(new_data).evaluate()
        
        return estimated_result.cpu().numpy(), gamma_matrix.cpu().numpy()
    #######################################################################
    # Baseline methods area ends here
    #######################################################################


    def init_fast_component(self, temp_data, condition_value=None):
        fast_component = {}
        # step 3.1: Compute the fast_X related term
        K_XX_t = self.k_x(self.pseudo_treatment, temp_data["treatment"]).evaluate()
        X_related = {'K_XX_t': K_XX_t}
        fast_component['X_related'] = X_related
        # step 3.2: Compute the fast_Z related term
        # step 3.3: Compute the fast_S related term
        K_SS_t = self.k_s(self.adjustment, temp_data["adjustment"]).evaluate()
        S_related = {'K_SS_t': K_SS_t}
        fast_component['S_related'] = S_related
        # step 3.4: compute the w-related term
        # step 3.5: Compute the K_W_t related term
        K_X_t = self.k_x(temp_data["treatment"], temp_data["treatment"]).evaluate()
        K_S_t = self.k_s(temp_data["adjustment"], temp_data["adjustment"]).evaluate()
        K_W_t = K_X_t * K_S_t
        tilde_K_W_t = K_W_t + torch.eye(K_W_t.shape[0]).to(self.device) * self.optimal_inner_gp_noise
        inverse_tilde_K_W_t = torch.linalg.inv(tilde_K_W_t)
        W_related = {'K_W_t': K_W_t, 'tilde_K_W_t': tilde_K_W_t, 'inverse_tilde_K_W_t': inverse_tilde_K_W_t}
        fast_component['W_related'] = W_related

        # to get the K_W_target
        K_X_target = self.k_x(self.pseudo_treatment, self.pseudo_treatment).evaluate()
        K_SS_target = self.k_s(self.adjustment, self.adjustment).evaluate()
        K_W_target = K_X_target * K_SS_target.mean()
        fast_component['K_W_target'] = K_W_target

        return fast_component


    def compute_variance(self, temp_data, new_data, fast_component, active_dataset):

        #Step 1: Prepare all the kernel objects and hyperparameters
        # Generate kernel object

        #NOTE: we cannot directly use the condition kernel from the GP model since
        #the act_dim is different. We need to create a new kernel object.
        # k_z_gp = self.model_gp.condition_kernel
        self.k_s = gpytorch.kernels.RBFKernel(
            lengthscale=self.optimal_adjustment_scale,
        ).to(self.device)
        self.k_x = self.model_gp.treatment_kernel

        #Step 2: Prepare the data
        # step 2.1: Prepare the full data
        self.treatment = self.treatment.to(self.device)
        treatment_interest = active_dataset.dataset.treatment_interest
        # treatment_interest = torch.tensor(treatment_interest).double().to(self.device)
        if self.treatment_type == "binary":
            if treatment_interest is None:
                self.pseudo_treatment = torch.cat([torch.zeros(1), torch.ones(1)], dim=0).reshape(-1,1).to(self.device)
            else:
                self.pseudo_treatment = torch.tensor(treatment_interest).double().reshape(-1,1).to(self.device)
        elif self.treatment_type == "discrete":
            if treatment_interest is None:
                self.pseudo_treatment = torch.unique(self.treatment).view(-1, 1)
            else:
                self.pseudo_treatment = torch.tensor(treatment_interest).double().reshape(-1,1).to(self.device)
        elif self.treatment_type == "continuous":
            assert treatment_interest is None
            self.pseudo_treatment = self.treatment

        self.adjustment = self.adjustment.to(self.device)
        # step 2.2: Prepare the temp data
        temp_data = copy.deepcopy(temp_data)
        temp_data['treatment'] = torch.tensor(temp_data['treatment']).double().to(self.device)
        temp_data['condition'] = torch.tensor(temp_data['condition']).double().to(self.device)
        temp_data['adjustment'] = torch.tensor(temp_data['adjustment']).double().to(self.device)
        temp_data['adjustment'] = torch.cat([temp_data['condition'], temp_data['adjustment']], dim=1)
        # step 2.3: Prepare the new data point
        new_data['treatment'] = torch.tensor(new_data['treatment']).double().to(self.device)
        new_data['condition'] = torch.tensor(new_data['condition']).double().to(self.device)
        new_data['adjustment'] = torch.tensor(new_data['adjustment']).double().to(self.device)
        new_data['adjustment'] = torch.cat([new_data['condition'],new_data['adjustment']], dim=1)

        #Step 3: Get the fast component term
        if len(fast_component) == 0: # if the fast component is empty
            fast_component = self.init_fast_component(temp_data)
        
        #Step 4: Compute the variance
        k_Xx_t1 = self.k_x(self.pseudo_treatment, new_data["treatment"]).evaluate()
        k_Ss_t1 = self.k_s(self.adjustment, new_data["adjustment"]).evaluate()
        K_XX_t1 = torch.concatenate([fast_component['X_related']['K_XX_t'], k_Xx_t1], dim=1)
        K_SS_t1 = torch.concatenate([fast_component['S_related']['K_SS_t'], k_Ss_t1], dim=1)
        n = self.treatment.shape[0]
        w_t1 = K_XX_t1 * (torch.ones(1, n, device=self.device, dtype=torch.double) / n @ K_SS_t1)

        K_X_tx_t1 = self.k_x(temp_data["treatment"], new_data["treatment"]).evaluate()
        K_S_ts_t1 = self.k_s(temp_data["adjustment"], new_data["adjustment"]).evaluate()
        k_W_t1 = K_X_tx_t1 * K_S_ts_t1

        k_x_t1x_t1 = self.k_x(new_data["treatment"], new_data["treatment"]).evaluate()
        k_s_t1s_t1 = self.k_s(new_data["adjustment"], new_data["adjustment"]).evaluate()
        k_w_t1 = k_x_t1x_t1 * k_s_t1s_t1

        inverse_tilde_K_W_t1 = expand_inverse_with_regularization(
            fast_component['W_related']['inverse_tilde_K_W_t'],
            k_W_t1,
            k_w_t1,
            self.optimal_inner_gp_noise
        ).to(self.device)

        var = w_t1 @ inverse_tilde_K_W_t1 @ w_t1.t()

        epsilon = 1e-6
        K_W_target = fast_component['K_W_target']
        cov = K_W_target - var + epsilon * torch.eye(var.size(0), device=var.device)

        return cov.detach().cpu().numpy(), fast_component


        # return -var.trace().detach().cpu().numpy(), fast_component


    def init_fast_cde_component(self, temp_data, target_data):
        fast_component = {}
        # condition_value_copy = torch.tensor(condition_value.detach().cpu().numpy()).double().reshape(-1, 1).to(self.device)
        # step 3.1: Compute the fast_X related term
        K_XX_t = self.k_x(target_data["treatment"], temp_data["treatment"]).evaluate()
        X_related = {'K_XX_t': K_XX_t}
        fast_component['X_related'] = X_related

        # step 3.3: Compute the fast_S related term
        K_SS_t = self.k_s(target_data["adjustment"], temp_data["adjustment"]).evaluate()
        S_related = {'K_SS_t': K_SS_t}
        fast_component['S_related'] = S_related

        w_t = K_XX_t * K_SS_t
        w_related = {'w_t': w_t}
        fast_component['w_related'] = w_related

        # step 3.5: Compute the K_W_t related term
        K_X_t = self.k_x(temp_data["treatment"], temp_data["treatment"]).evaluate()
        K_S_t = self.k_s(temp_data["adjustment"], temp_data["adjustment"]).evaluate()
        K_W_t = K_X_t * K_S_t

        tilde_K_W_t = K_W_t + torch.eye(K_W_t.shape[0]).to(self.device) * self.optimal_inner_gp_noise
        inverse_tilde_K_W_t = torch.linalg.inv(tilde_K_W_t)
        W_related = {'K_W_t': K_W_t, 'tilde_K_W_t': tilde_K_W_t, 'inverse_tilde_K_W_t': inverse_tilde_K_W_t}
        fast_component['W_related'] = W_related


        K_X_target = self.k_x(target_data["treatment"], target_data["treatment"]).evaluate()
        K_S_target = self.k_s(target_data["adjustment"], target_data["adjustment"]).evaluate()
        K_W_target = K_X_target * K_S_target
        fast_component['K_W_target'] = K_W_target

        return fast_component

    
    def compute_cde_variance(self, temp_data, new_data, fast_component, target_data, sampling_size):

        #Step 1: Prepare all the kernel objects and hyperparameters
        self.k_s = gpytorch.kernels.RBFKernel(
            lengthscale=self.optimal_adjustment_scale,
            ).to(self.device)

        self.k_x = self.model_gp.treatment_kernel

        #Step 2: Prepare the data
        # step 2.1: Prepare the full data
        target_data = copy.deepcopy(target_data)
        target_data['treatment'] = torch.tensor(target_data['treatment']).double().to(self.device)
        target_data['condition'] = torch.tensor(target_data['condition']).double().to(self.device)
        target_data['adjustment'] = torch.tensor(target_data['adjustment']).double().to(self.device)
        target_data['adjustment'] = torch.cat([target_data['condition'], target_data['adjustment']], dim=1)
        # step 2.2: Prepare the temp data
        temp_data = copy.deepcopy(temp_data)
        temp_data['treatment'] = torch.tensor(temp_data['treatment']).double().to(self.device)
        temp_data['condition'] = torch.tensor(temp_data['condition']).double().to(self.device)
        temp_data['adjustment'] = torch.tensor(temp_data['adjustment']).double().to(self.device)
        temp_data['adjustment'] = torch.cat([temp_data['condition'], temp_data['adjustment']], dim=1)
        # step 2.3: Prepare the new data point
        new_data['treatment'] = torch.tensor(new_data['treatment']).double().to(self.device)
        new_data['condition'] = torch.tensor(new_data['condition']).double().to(self.device)
        new_data['adjustment'] = torch.tensor(new_data['adjustment']).double().to(self.device)
        new_data['adjustment'] = torch.cat([new_data['condition'],new_data['adjustment']], dim=1)

        #Step 3: Get the fast component term
        if len(fast_component) == 0: # if the fast component is empty
            fast_component = self.init_fast_cde_component(temp_data, target_data)
        
        #Step 4: Compute the variance
        k_Xx_t1 = self.k_x(target_data["treatment"], new_data["treatment"]).evaluate()
        k_Ss_t1 = self.k_s(target_data["adjustment"], new_data["adjustment"]).evaluate()
        k_w_Xw_t1 = k_Xx_t1 * k_Ss_t1
        w_t1 = torch.concatenate([fast_component['w_related']['w_t'], k_w_Xw_t1], dim=1)

        K_X_tx_t1 = self.k_x(temp_data["treatment"], new_data["treatment"]).evaluate()
        K_S_ts_t1 = self.k_s(temp_data["adjustment"], new_data["adjustment"]).evaluate()
        k_W_t1 = K_X_tx_t1 * K_S_ts_t1

        k_x_t1x_t1 = self.k_x(new_data["treatment"], new_data["treatment"]).evaluate()
        k_s_t1s_t1 = self.k_s(new_data["adjustment"], new_data["adjustment"]).evaluate()
        k_w_t1 = k_x_t1x_t1 * k_s_t1s_t1

        inverse_tilde_K_W_t1 = expand_inverse_with_regularization(
            fast_component['W_related']['inverse_tilde_K_W_t'],
            k_W_t1,
            k_w_t1,
            self.optimal_inner_gp_noise
        ).to(self.device)

        var = w_t1 @ inverse_tilde_K_W_t1 @ w_t1.t()

        # return -var.trace().detach().cpu().numpy(), fast_component
        epsilon = 1e-6
        K_W_target = fast_component['K_W_target']
        cov = K_W_target - var + epsilon * torch.eye(var.shape[0]).to(self.device)

        cov = cov.view(cov.size(0) // sampling_size, sampling_size, -1, sampling_size).mean(dim=(1, 3))

        return cov.detach().cpu().numpy(), fast_component
        # return torch.logdet(var).detach().cpu().numpy(), fast_component
    


    def predict(self, ds_test):
        self.model_gp.eval()
        self.likelihood_gp.eval()

        treatment = torch.tensor(ds_test.inputs_set["treatment"]).double().to(self.device)
        targets = torch.tensor(ds_test.targets).double().to(self.device)
        inputs = torch.tensor(ds_test.inputs).double().to(self.device)

        if hasattr(ds_test, 'treatment_interest'):
            treatment_interest = ds_test.treatment_interest
        else:
            treatment_interest = ds_test.dataset.treatment_interest

        if treatment_interest is not None:
            mask = (treatment == treatment_interest).squeeze()
            treatment = treatment[mask]
            targets = targets[mask]
            inputs = inputs[mask]

        # Clustering the same treatment group
        treatment_unique = torch.unique(treatment).reshape(-1,1)

        amse_list = []
        for t in treatment_unique:
            idx_temp_t = torch.where(treatment == t)[0]
            # Predict the potential outcomes
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                estimated_result = self.likelihood_gp(self.model_gp(inputs[idx_temp_t])).mean.detach()
                # get the differnce between the potential outcomes
                se = (torch.mean(estimated_result) - torch.mean(targets[idx_temp_t]))**2
                amse_list.append(se)
        
        # return torch.mean(torch.tensor(amse_list)).cpu().numpy()
        return torch.sqrt(torch.mean(torch.tensor(amse_list))).cpu().numpy()







############################################################################################################
########################                 ATT           #####################################################
############################################################################################################
class IMP_ATT(ABC):
    def __init__(
        self,
        job_dir,
        task_type,
        treatment_type,
        active_dataset,
        tune_dataset,
        learning_rate,
        epochs_gp,
        epochs_cme,
        learn_cme,
        patience,
        num_workers,
        device,
        seed,
    ):
        super(IMP_ATT, self).__init__()
        self.job_dir = job_dir
        self.task_type = task_type
        self.treatment_type = treatment_type
        self.active_dataset = active_dataset
        self.tune_dataset = tune_dataset
        self.learning_rate = learning_rate
        self.best_loss = 1e7
        self.patience = patience
        self.counter = 0
        self.epochs_gp = epochs_gp
        self.epochs_cme = epochs_cme
        self.learn_cme = learn_cme
        self.num_workers = num_workers
        self.device = device
        self.seed = seed

        #NOTE: we only consider the binary and discrete treatment type. Since the continuous treatment type
        #is not easy to condition on.
        assert self.treatment_type in ["binary", "discrete"]

        #######################################################################
        # The inner GP model related stuff
        #NOTE: when you are using the gpytorch model, plase pay attention to the dimension of the input data.
        
        self.inputs = torch.tensor(active_dataset.extract_active_data()[0]).double()
        self.targets = torch.tensor(active_dataset.extract_active_data()[1]).double().squeeze()
        self.inputs_val = torch.tensor(tune_dataset.inputs).double()
        self.targets_val = torch.tensor(tune_dataset.targets).double().squeeze()

        if hasattr(active_dataset.dataset, 'dim_condition'):
            dim_adjustment = active_dataset.dataset.dim_adjustment + active_dataset.dataset.dim_condition
        else:
            dim_adjustment = active_dataset.dataset.dim_adjustment

        self.likelihood_gp = gpytorch.likelihoods.GaussianLikelihood()
        self.model_gp = gaussian_process.ExactMultiInputGPModel_ate_att(
            train_x=self.inputs,
            train_y=self.targets,
            task_type=self.task_type,
            treatment_type=self.treatment_type,
            dim_treatment=self.active_dataset.dataset.dim_treatment,
            dim_adjustment=dim_adjustment,
            likelihood=self.likelihood_gp,
        ).to(self.device)

        self.optimizer_gp = optim.Adam(self.model_gp.parameters(),
                                    lr=self.learning_rate)
        self.mll_gp = mlls.ExactMarginalLogLikelihood(self.likelihood_gp, self.model_gp)
        #######################################################################


        #######################################################################
        # The inner CME model related stuff
        # Data preparation
        self.treatment = torch.tensor(self.active_dataset.dataset.inputs_set["treatment"]).double()
        self.condition = torch.tensor(self.active_dataset.dataset.inputs_set["condition"]).double()
        self.adjustment = torch.tensor(self.active_dataset.dataset.inputs_set["adjustment"]).double()
        self.adjustment = torch.cat([self.condition, self.adjustment], dim=1)

        #NOTE: only for the discrete case, we need to train the CME model.
        self.lambda_cme = torch.Tensor([np.sqrt(0.01)]).double()
        self.model_cme = CME.CME_learner(
            x=self.treatment,
            y=self.adjustment,
            lambda_init=self.lambda_cme,
            device=self.device,
        ).to(self.device)

        self.optimizer_cme = optim.Adam(self.model_cme.parameters(),
                                    lr=self.learning_rate)
        self.loss_cme = CME.nll
        #######################################################################

    def fit_gp(self, exp_mode):

        for _ in range(self.epochs_gp):

            self.model_gp.train()
            self.likelihood_gp.train()

            inputs, targets = self.inputs.to(self.device), self.targets.to(self.device)
            self.optimizer_gp.zero_grad()
            output = self.model_gp(inputs)
            loss = -self.mll_gp(output, targets)
            loss.backward()
            self.optimizer_gp.step()

            # validation loss
            if _ % 10 == 0:
                self.model_gp.eval()
                self.likelihood_gp.eval()
                with torch.no_grad(), gpytorch.settings.fast_pred_var():

                    inputs_val, targets_val = self.inputs_val.to(self.device), self.targets_val.to(self.device)
                    output_val = self.likelihood_gp(self.model_gp(inputs_val))
                    loss_val = -self.mll_gp(output_val, targets_val)
                    # Early stopping after 300 epochs
                    if loss_val < self.best_loss:
                        self.best_loss = loss_val
                        self.counter = 0
                    else:
                        if _ > 200:
                            self.counter += 1
                    if self.counter > self.patience:
                        break
        
        self.model_gp.eval()
        self.likelihood_gp.eval()

        self.optimal_adjustment_scale = self.model_gp.adjustment_kernel.lengthscale.detach()
        if self.treatment_type == "discrete":
            self.optimal_treatment_scale = self.model_gp.treatment_kernel.lengthscale.detach()
        self.optimal_inner_gp_noise = self.model_gp.likelihood.noise.detach()

    def fit_cme(self, exp_mode):

        self.model_cme.l_y.raw_lengthscale.requires_grad = False
        self.model_cme.l_y.lengthscale = self.optimal_adjustment_scale

        self.model_cme.train()
        for _ in tqdm(range(self.epochs_cme)):
            self.optimizer_cme.zero_grad()
            train_K_lambda, train_L = self.model_cme()
            loss = self.loss_cme(train_K_lambda, train_L)
            loss.backward()
            self.optimizer_cme.step()
    
        self.model_cme.eval()
        self.optimal_lambda_cme = self.model_cme.lmda.detach()
        self.optimal_cme_treatment_scale = self.model_cme.k_x.lengthscale.detach()
        
    def save(self,):
        torch.save(self.model_gp.state_dict(), self.job_dir / "model_gp_best.pt")
        torch.save(self.model_cme.state_dict(), self.job_dir / "model_cme_best.pt")
    
    def load(self,):
        self.model_gp.load_state_dict(torch.load(self.job_dir / "model_gp_best.pt"))
        self.model_cme.load_state_dict(torch.load(self.job_dir / "model_cme_best.pt"))

    def fit(self, exp_mode):
        self.fit_gp(exp_mode=exp_mode)

        if self.learn_cme:
            self.fit_cme(exp_mode=exp_mode)
        self.save()

    #######################################################################
    # Baseline methods area starts here
    #######################################################################
    def init_fast_naive_component(self, temp_data, condition_value=None):
        fast_component = {}
        K_XX_t = self.k_x(self.treatment, temp_data["treatment"]).evaluate()
        X_related = {'K_XX_t': K_XX_t}
        fast_component['X_related'] = X_related

        # step 3.3: Compute the fast_S related term
        K_SS_t = self.k_s(self.adjustment, temp_data["adjustment"]).evaluate()
        S_related = {'K_SS_t': K_SS_t}
        fast_component['S_related'] = S_related

        w_t = K_XX_t * K_SS_t
        w_related = {'w_t': w_t}
        fast_component['w_related'] = w_related

        # step 3.5: Compute the K_W_t related term
        K_X_t = self.k_x(temp_data["treatment"], temp_data["treatment"]).evaluate()
        K_S_t = self.k_s(temp_data["adjustment"], temp_data["adjustment"]).evaluate()
        K_W_t = K_X_t * K_S_t

        tilde_K_W_t = K_W_t + torch.eye(K_W_t.shape[0]).to(self.device) * self.optimal_inner_gp_noise
        inverse_tilde_K_W_t = torch.linalg.inv(tilde_K_W_t)
        W_related = {'K_W_t': K_W_t, 'tilde_K_W_t': tilde_K_W_t, 'inverse_tilde_K_W_t': inverse_tilde_K_W_t}
        fast_component['W_related'] = W_related

        return fast_component

    
    def compute_naive_variance(self, temp_data, new_data, fast_component, active_dataset):

        #Step 1: Prepare all the kernel objects and hyperparameters
        self.k_s = gpytorch.kernels.RBFKernel(
            lengthscale=self.optimal_adjustment_scale,
            ).to(self.device)

        self.k_x = self.model_gp.treatment_kernel

        #Step 2: Prepare the data
        # step 2.1: Prepare the full data
        self.treatment = self.treatment.to(self.device)
        self.adjustment = self.adjustment.to(self.device)
        # step 2.2: Prepare the temp data
        temp_data = copy.deepcopy(temp_data)
        temp_data['treatment'] = torch.tensor(temp_data['treatment']).double().to(self.device)
        temp_data['condition'] = torch.tensor(temp_data['condition']).double().to(self.device)
        temp_data['adjustment'] = torch.tensor(temp_data['adjustment']).double().to(self.device)
        temp_data['adjustment'] = torch.cat([temp_data['condition'], temp_data['adjustment']], dim=1)
        # step 2.3: Prepare the new data point
        new_data['treatment'] = torch.tensor(new_data['treatment']).double().to(self.device)
        new_data['condition'] = torch.tensor(new_data['condition']).double().to(self.device)
        new_data['adjustment'] = torch.tensor(new_data['adjustment']).double().to(self.device)
        new_data['adjustment'] = torch.cat([new_data['condition'],new_data['adjustment']], dim=1)

        #Step 3: Get the fast component term
        if len(fast_component) == 0: # if the fast component is empty
            fast_component = self.init_fast_naive_component(temp_data)
        
        #Step 4: Compute the variance
        k_Xx_t1 = self.k_x(self.treatment, new_data["treatment"]).evaluate()
        k_Ss_t1 = self.k_s(self.adjustment, new_data["adjustment"]).evaluate()
        k_w_Xw_t1 = k_Xx_t1 * k_Ss_t1
        w_t1 = torch.concatenate([fast_component['w_related']['w_t'], k_w_Xw_t1], dim=1)

        K_X_tx_t1 = self.k_x(temp_data["treatment"], new_data["treatment"]).evaluate()
        K_S_ts_t1 = self.k_s(temp_data["adjustment"], new_data["adjustment"]).evaluate()
        k_W_t1 = K_X_tx_t1 * K_S_ts_t1

        k_x_t1x_t1 = self.k_x(new_data["treatment"], new_data["treatment"]).evaluate()
        k_s_t1s_t1 = self.k_s(new_data["adjustment"], new_data["adjustment"]).evaluate()
        k_w_t1 = k_x_t1x_t1 * k_s_t1s_t1

        inverse_tilde_K_W_t1 = expand_inverse_with_regularization(
            fast_component['W_related']['inverse_tilde_K_W_t'],
            k_W_t1,
            k_w_t1,
            self.optimal_inner_gp_noise
        ).to(self.device)

        var = w_t1 @ inverse_tilde_K_W_t1 @ w_t1.t()

        return -var.trace().detach().cpu().numpy(), fast_component
    
    def compute_bald_variance(self, new_data):
        self.model_gp.eval()
        self.likelihood_gp.eval()

        new_data = torch.tensor(new_data).double().to(self.device)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            estimated_result = self.likelihood_gp(self.model_gp(new_data)).variance.detach()
            gamma_matrix = self.model_gp.covar_module(new_data).evaluate()
        
        return estimated_result.cpu().numpy(), gamma_matrix.cpu().numpy()
    #######################################################################
    # Baseline methods area ends here
    #######################################################################
    


    def init_fast_component(self, temp_data, treatment_value=None):
        fast_component = {}
        # step 3.1: Compute the fast_X related term
        # if self.treatment_type == "discrete":
        #     K_XX_t = self.k_x(self.treatment, temp_data["treatment"]).evaluate()
        # elif self.treatment_type == "binary":
        #     K_XX_t = self.k_x(self.pseudo_treatment, temp_data["treatment"]).evaluate()
        # else:
        #     raise ValueError("The treatment type is not supported.")
        K_XX_t = self.k_x(self.pseudo_treatment, temp_data["treatment"]).evaluate()
        X_related = {'K_XX_t': K_XX_t}
        fast_component['X_related'] = X_related
        # step 3.2: Compute the fast_Z related term
        # step 3.3: Compute the fast_S related term
        if self.treatment_type == "discrete":
            # fixed_left = self.k_x_cme(treatment_value, self.treatment).evaluate() @ torch.linalg.inv(
            #     self.k_x_cme(self.treatment, self.treatment).add_diagonal(self.optimal_lambda_cme).evaluate())
            #NOTE: the above code sometimes will raise the error: AttributeError: 'LazyEvaluatedKernelTensor' object has no attribute 'add_diagonal'
            fixed_left = self.k_x_cme(treatment_value, self.treatment).evaluate() @ torch.linalg.inv(
                self.k_x_cme(self.treatment, self.treatment).evaluate() + torch.eye(self.treatment.shape[0]).to(self.device) * self.optimal_lambda_cme)
            K_SS_t = self.k_s(self.adjustment, temp_data["adjustment"]).evaluate()
            S_related = {'fixed_left': fixed_left, 'K_SS_t': K_SS_t}
            fast_component['S_related'] = S_related
            # step 3.4: compute the w-related term
            w_t = K_XX_t * (fixed_left @ K_SS_t)
            w_related = {'w_t': w_t}
        elif self.treatment_type == "binary":
            cared_idx = torch.where(self.treatment == treatment_value)[0]
            cared_idx_len = cared_idx.shape[0]
            K_SS_t = self.k_s(self.adjustment[cared_idx], temp_data["adjustment"]).evaluate()
            S_related = {'K_SS_t': K_SS_t}
            fast_component['S_related'] = S_related
            # step 3.4: compute the w-related term
            w_related = {'cared_idx_len': cared_idx_len, 'cared_idx': cared_idx}
        fast_component['w_related'] = w_related
        # step 3.5: Compute the K_W_t related term
        K_X_t = self.k_x(temp_data["treatment"], temp_data["treatment"]).evaluate()
        K_S_t = self.k_s(temp_data["adjustment"], temp_data["adjustment"]).evaluate()
        K_W_t = K_X_t * K_S_t
        tilde_K_W_t = K_W_t + torch.eye(K_W_t.shape[0]).to(self.device) * self.optimal_inner_gp_noise
        inverse_tilde_K_W_t = torch.linalg.inv(tilde_K_W_t)
        W_related = {'K_W_t': K_W_t, 'tilde_K_W_t': tilde_K_W_t, 'inverse_tilde_K_W_t': inverse_tilde_K_W_t}
        fast_component['W_related'] = W_related

        # to get the K_W_target
        if self.treatment_type == "discrete":
            K_X_target = self.k_x(self.pseudo_treatment, self.pseudo_treatment).evaluate()
            K_SS = self.k_s(self.adjustment, self.adjustment).evaluate()
            K_W_target = K_X_target * (fixed_left @ K_SS @ fixed_left.t())
        elif self.treatment_type == "binary":
            K_X_target = self.k_x(self.pseudo_treatment, self.pseudo_treatment).evaluate()
            K_SS_target = self.k_s(self.adjustment[cared_idx], self.adjustment[cared_idx]).evaluate()
            K_W_target = K_X_target * K_SS_target.mean()
        else:
            raise ValueError("The treatment type is not supported.")
        
        fast_component['K_W_target'] = K_W_target

        return fast_component


    def compute_variance(self, temp_data, new_data, fast_component, active_dataset):

        treatment_value = active_dataset.dataset.treatment_value
        assert treatment_value is not None

        #Step 1: Prepare all the kernel objects and hyperparameters
        # Generate kernel object
        if self.treatment_type == "discrete":
            self.k_x_cme = self.model_cme.k_x
        # self.k_s = self.model_cme.l_y

        #NOTE: we cannot directly use the condition kernel from the GP model since
        #the act_dim is different. We need to create a new kernel object.
        # k_z_gp = self.model_gp.condition_kernel
        self.k_s = gpytorch.kernels.RBFKernel(
            lengthscale=self.optimal_adjustment_scale,
        ).to(self.device)
        self.k_x = self.model_gp.treatment_kernel

        #Step 2: Prepare the data
        # step 2.1: Prepare the full data
        self.treatment = self.treatment.to(self.device)
        treatment_interest = active_dataset.dataset.treatment_interest
        # treatment_interest = torch.tensor(treatment_interest).double().to(self.device)
        if self.treatment_type == "binary":
            if treatment_interest is None:
                self.pseudo_treatment = torch.cat([torch.zeros(1), torch.ones(1)], dim=0).reshape(-1,1).to(self.device)
            else:
                self.pseudo_treatment = torch.tensor(treatment_interest).double().reshape(-1,1).to(self.device)
        else:
            if treatment_interest is None:
                self.pseudo_treatment = torch.unique(self.treatment).view(-1, 1)
            else:
                self.pseudo_treatment = torch.tensor(treatment_interest).double().reshape(-1,1).to(self.device)
        self.adjustment = self.adjustment.to(self.device)
        # step 2.2: Prepare the temp data
        temp_data = copy.deepcopy(temp_data)
        temp_data['treatment'] = torch.tensor(temp_data['treatment']).double().to(self.device)
        temp_data['condition'] = torch.tensor(temp_data['condition']).double().to(self.device)
        temp_data['adjustment'] = torch.tensor(temp_data['adjustment']).double().to(self.device)
        temp_data['adjustment'] = torch.cat([temp_data['condition'], temp_data['adjustment']], dim=1)
        treatment_value = torch.tensor(treatment_value).double().reshape(1,1).to(self.device)
        # step 2.3: Prepare the new data point
        new_data['treatment'] = torch.tensor(new_data['treatment']).double().to(self.device)
        new_data['condition'] = torch.tensor(new_data['condition']).double().to(self.device)
        new_data['adjustment'] = torch.tensor(new_data['adjustment']).double().to(self.device)
        new_data['adjustment'] = torch.cat([new_data['condition'],new_data['adjustment']], dim=1)

        #Step 3: Get the fast component term
        if len(fast_component) == 0: # if the fast component is empty
            fast_component = self.init_fast_component(temp_data, treatment_value)
        
        #Step 4: Compute the variance
        # if self.treatment_type == "discrete":
        #     k_Xx_t1 = self.k_x(self.treatment, new_data["treatment"]).evaluate()
        # elif self.treatment_type == "binary":
        #     k_Xx_t1 = self.k_x(self.pseudo_treatment, new_data["treatment"]).evaluate()
        # else:
        #     raise ValueError("The treatment type is not supported.")
        k_Xx_t1 = self.k_x(self.pseudo_treatment, new_data["treatment"]).evaluate()
        K_XX_t1 = torch.concatenate([fast_component['X_related']['K_XX_t'], k_Xx_t1], dim=1)
        if self.treatment_type == "discrete":
            s_part_s_t1 = fast_component['S_related']['fixed_left'] @ self.k_s(self.adjustment, new_data["adjustment"]).evaluate()
            k_w_Xw_t1 = k_Xx_t1 * s_part_s_t1
            w_t1 = torch.concatenate([fast_component['w_related']['w_t'], k_w_Xw_t1], dim=1)
        elif self.treatment_type == "binary":
            cared_idx = fast_component['w_related']['cared_idx']
            cared_idx_len = fast_component['w_related']['cared_idx_len']
            k_s_t1 = self.k_s(self.adjustment[cared_idx], new_data["adjustment"]).evaluate()
            K_SS_t1 = torch.cat([fast_component['S_related']['K_SS_t'], k_s_t1], dim=1)
            w_t1 = K_XX_t1 * (torch.ones(1, cared_idx_len, device=self.device, dtype=torch.double) / cared_idx_len @ K_SS_t1)
        else:
            raise ValueError("The treatment type is not supported.")

        K_X_tx_t1 = self.k_x(temp_data["treatment"], new_data["treatment"]).evaluate()
        K_S_ts_t1 = self.k_s(temp_data["adjustment"], new_data["adjustment"]).evaluate()
        k_W_t1 = K_X_tx_t1 * K_S_ts_t1

        k_x_t1x_t1 = self.k_x(new_data["treatment"], new_data["treatment"]).evaluate()
        k_s_t1s_t1 = self.k_s(new_data["adjustment"], new_data["adjustment"]).evaluate()
        k_w_t1 = k_x_t1x_t1 * k_s_t1s_t1

        inverse_tilde_K_W_t1 = expand_inverse_with_regularization(
            fast_component['W_related']['inverse_tilde_K_W_t'],
            k_W_t1,
            k_w_t1,
            self.optimal_inner_gp_noise
        ).to(self.device)

        var = w_t1 @ inverse_tilde_K_W_t1 @ w_t1.t()

        epsilon = 1e-6
        K_W_target = fast_component['K_W_target']
        cov = K_W_target - var + epsilon * torch.eye(var.shape[0]).to(self.device)

        return cov.detach().cpu().numpy(), fast_component

        # return -var.trace().detach().cpu().numpy(), fast_component
    


    def init_fast_cde_component(self, temp_data, target_data):
        fast_component = {}
        # condition_value_copy = torch.tensor(condition_value.detach().cpu().numpy()).double().reshape(-1, 1).to(self.device)
        # step 3.1: Compute the fast_X related term
        K_XX_t = self.k_x(target_data["treatment"], temp_data["treatment"]).evaluate()
        X_related = {'K_XX_t': K_XX_t}
        fast_component['X_related'] = X_related

        # step 3.3: Compute the fast_S related term
        K_SS_t = self.k_s(target_data["adjustment"], temp_data["adjustment"]).evaluate()
        S_related = {'K_SS_t': K_SS_t}
        fast_component['S_related'] = S_related

        w_t = K_XX_t * K_SS_t
        w_related = {'w_t': w_t}
        fast_component['w_related'] = w_related

        # step 3.5: Compute the K_W_t related term
        K_X_t = self.k_x(temp_data["treatment"], temp_data["treatment"]).evaluate()
        K_S_t = self.k_s(temp_data["adjustment"], temp_data["adjustment"]).evaluate()
        K_W_t = K_X_t * K_S_t

        tilde_K_W_t = K_W_t + torch.eye(K_W_t.shape[0]).to(self.device) * self.optimal_inner_gp_noise
        inverse_tilde_K_W_t = torch.linalg.inv(tilde_K_W_t)
        W_related = {'K_W_t': K_W_t, 'tilde_K_W_t': tilde_K_W_t, 'inverse_tilde_K_W_t': inverse_tilde_K_W_t}
        fast_component['W_related'] = W_related

        K_X_target = self.k_x(target_data["treatment"], target_data["treatment"]).evaluate()
        K_S_target = self.k_s(target_data["adjustment"], target_data["adjustment"]).evaluate()
        K_W_target = K_X_target * K_S_target
        fast_component['K_W_target'] = K_W_target

        return fast_component

    
    def compute_cde_variance(self, temp_data, new_data, fast_component, target_data, sampling_size):

        #Step 1: Prepare all the kernel objects and hyperparameters
        self.k_s = gpytorch.kernels.RBFKernel(
            lengthscale=self.optimal_adjustment_scale,
            ).to(self.device)

        self.k_x = self.model_gp.treatment_kernel

        #Step 2: Prepare the data
        # step 2.1: Prepare the full data
        target_data = copy.deepcopy(target_data)
        target_data['treatment'] = torch.tensor(target_data['treatment']).double().to(self.device)
        target_data['condition'] = torch.tensor(target_data['condition']).double().to(self.device)
        target_data['adjustment'] = torch.tensor(target_data['adjustment']).double().to(self.device)
        target_data['adjustment'] = torch.cat([target_data['condition'], target_data['adjustment']], dim=1)
        # step 2.2: Prepare the temp data
        temp_data = copy.deepcopy(temp_data)
        temp_data['treatment'] = torch.tensor(temp_data['treatment']).double().to(self.device)
        temp_data['condition'] = torch.tensor(temp_data['condition']).double().to(self.device)
        temp_data['adjustment'] = torch.tensor(temp_data['adjustment']).double().to(self.device)
        temp_data['adjustment'] = torch.cat([temp_data['condition'], temp_data['adjustment']], dim=1)
        # step 2.3: Prepare the new data point
        new_data['treatment'] = torch.tensor(new_data['treatment']).double().to(self.device)
        new_data['condition'] = torch.tensor(new_data['condition']).double().to(self.device)
        new_data['adjustment'] = torch.tensor(new_data['adjustment']).double().to(self.device)
        new_data['adjustment'] = torch.cat([new_data['condition'],new_data['adjustment']], dim=1)

        #Step 3: Get the fast component term
        if len(fast_component) == 0: # if the fast component is empty
            fast_component = self.init_fast_cde_component(temp_data, target_data)
        
        #Step 4: Compute the variance
        k_Xx_t1 = self.k_x(target_data["treatment"], new_data["treatment"]).evaluate()
        k_Ss_t1 = self.k_s(target_data["adjustment"], new_data["adjustment"]).evaluate()
        k_w_Xw_t1 = k_Xx_t1 * k_Ss_t1
        w_t1 = torch.concatenate([fast_component['w_related']['w_t'], k_w_Xw_t1], dim=1)

        K_X_tx_t1 = self.k_x(temp_data["treatment"], new_data["treatment"]).evaluate()
        K_S_ts_t1 = self.k_s(temp_data["adjustment"], new_data["adjustment"]).evaluate()
        k_W_t1 = K_X_tx_t1 * K_S_ts_t1

        k_x_t1x_t1 = self.k_x(new_data["treatment"], new_data["treatment"]).evaluate()
        k_s_t1s_t1 = self.k_s(new_data["adjustment"], new_data["adjustment"]).evaluate()
        k_w_t1 = k_x_t1x_t1 * k_s_t1s_t1

        inverse_tilde_K_W_t1 = expand_inverse_with_regularization(
            fast_component['W_related']['inverse_tilde_K_W_t'],
            k_W_t1,
            k_w_t1,
            self.optimal_inner_gp_noise
        ).to(self.device)

        var = w_t1 @ inverse_tilde_K_W_t1 @ w_t1.t()

        # return -var.trace().detach().cpu().numpy(), fast_component
        epsilon = 1e-6
        K_W_target = fast_component['K_W_target']
        cov = K_W_target - var + epsilon * torch.eye(var.shape[0]).to(self.device)

        cov = cov.view(cov.size(0) // sampling_size, sampling_size, -1, sampling_size).mean(dim=(1, 3))

        return cov.detach().cpu().numpy(), fast_component

        # return torch.logdet(var).detach().cpu().numpy(), fast_component


    
    def predict(self, ds_test):
        self.model_gp.eval()
        self.likelihood_gp.eval()
        self.model_cme.eval()

        treatment = torch.tensor(ds_test.inputs_set["treatment"]).double().to(self.device)
        targets = torch.tensor(ds_test.targets).double().to(self.device)
        inputs = torch.tensor(ds_test.inputs).double().to(self.device)

        if hasattr(ds_test, 'treatment_interest'):
            treatment_interest = ds_test.treatment_interest
        else:
            treatment_interest = ds_test.dataset.treatment_interest

        if treatment_interest is not None:
            mask = (treatment == treatment_interest).squeeze()
            treatment = treatment[mask]
            targets = targets[mask]
            inputs = inputs[mask]

        # Clustering the same treatment group
        treatment_unique = torch.unique(treatment).reshape(-1,1)

        amse_list = []
        for t in treatment_unique:
            idx_temp_t = torch.where(treatment == t)[0]
            # Predict the potential outcomes
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                estimated_result = self.likelihood_gp(self.model_gp(inputs[idx_temp_t])).mean.detach()
                # get the differnce between the potential outcomes
                se = (torch.mean(estimated_result) - torch.mean(targets[idx_temp_t]))**2
                amse_list.append(se)
        
        # return torch.mean(torch.tensor(amse_list)).cpu().numpy()
        return torch.sqrt(torch.mean(torch.tensor(amse_list))).cpu().numpy()
    