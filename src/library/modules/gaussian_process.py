import torch
import gpytorch

class DeltaKernel(gpytorch.kernels.Kernel):
    def __init__(self, **kwargs):
        super(DeltaKernel, self).__init__(**kwargs)
        # self.is_stationary = True

    def forward(self, x1, x2, diag=False, **params):
        # Ensure inputs are of compatible shapes
        if x1.shape[-1] != x2.shape[-1]:
            raise ValueError("Input tensors must have the same last dimension size.")

        # Calculate the delta function: 1 if x1[i] == x2[j], 0 otherwise
        diff = x1.unsqueeze(-2) - x2.unsqueeze(-3)
        delta = torch.all(diff == 0, dim=-1).float()

        if diag:
            return delta.diag()
        return delta


class ExactMultiInputGPModel_ate_att(gpytorch.models.ExactGP):
    def __init__(self,train_x,
                train_y,
                likelihood,
                task_type,
                treatment_type,
                dim_treatment,
                dim_adjustment):
        super(ExactMultiInputGPModel_ate_att, self).__init__(train_x,
                                                    train_y,
                                                    likelihood)

        self.task_type = task_type
        self.treatment_type = treatment_type
        self.dim_treatment = dim_treatment
        self.dim_adjustment = dim_adjustment

        assert self.task_type in ["ate", "att", "ds"]
        assert self.treatment_type in ["binary", "continuous", "discrete"]
        assert self.dim_treatment == 1

        if self.treatment_type in ["continuous", "discrete"]:
            self.treatment_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=1,
                                                            active_dims=torch.tensor([0]))
        else:
            self.treatment_kernel = DeltaKernel(active_dims=torch.tensor([0]))

        # we combine the condition and adjustment features into the adjustment kernel
        active_dims_adjustment = list(range(1, self.dim_adjustment + 1))
        self.adjustment_kernel = gpytorch.kernels.RBFKernel(active_dims=active_dims_adjustment,
                                                            ard_num_dims=self.dim_adjustment)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = self.treatment_kernel * self.adjustment_kernel

    def forward(self, x):

        # check if the input has the right shape
        assert x.shape[1] == self.dim_treatment + self.dim_adjustment

        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        

class ExactMultiInputGPModel_cate(gpytorch.models.ExactGP):
    def __init__(self,train_x,
                train_y,
                likelihood,
                task_type,
                condition_type,
                treatment_type,
                dim_treatment,
                dim_condition,
                dim_adjustment,
                kernel_type):
        
        super(ExactMultiInputGPModel_cate, self).__init__(train_x,
                                                    train_y,
                                                    likelihood)

        self.task_type = task_type
        self.condition_type = condition_type
        self.treatment_type = treatment_type
        self.dim_treatment = dim_treatment
        self.dim_condition = dim_condition
        self.dim_adjustment = dim_adjustment
        self.kernel_type = kernel_type

        assert self.task_type in ["cate"]
        assert self.treatment_type in ["binary", "discrete", "continuous"]
        assert self.dim_treatment == 1
        assert self.dim_condition == 1

        if self.treatment_type in ["continuous", "discrete"]:
            if self.kernel_type == "rbf":
                self.treatment_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=1,
                                                                active_dims=torch.tensor([0]))
            elif self.kernel_type == "matern":
                self.treatment_kernel = gpytorch.kernels.MaternKernel(ard_num_dims=1,
                                                                active_dims=torch.tensor([0]))
            elif self.kernel_type == "rq":
                self.treatment_kernel = gpytorch.kernels.RQKernel(ard_num_dims=1,
                                                                active_dims=torch.tensor([0]))
            else:
                raise ValueError("Kernel type not recognized.")
        else:
            self.treatment_kernel = DeltaKernel(active_dims=torch.tensor([0]))
        
        if self.condition_type in ["continuous", "discrete"]:
            if self.kernel_type == "rbf":
                self.condition_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=1,
                                                                active_dims=torch.tensor([1]))
            elif self.kernel_type == "matern":
                self.condition_kernel = gpytorch.kernels.MaternKernel(ard_num_dims=1,
                                                                active_dims=torch.tensor([1]))
            elif self.kernel_type == "rq":
                self.condition_kernel = gpytorch.kernels.RQKernel(ard_num_dims=1,
                                                                active_dims=torch.tensor([1]))
            else:
                raise ValueError("Kernel type not recognized.")

        elif self.condition_type == "binary":
            self.condition_kernel = DeltaKernel(active_dims=torch.tensor([1]))
        else:
            raise ValueError("Condition type not recognized.")

        active_dims_adjustment = list(range(2, self.dim_condition + self.dim_adjustment + 1))
        # self.adjustment_kernel = gpytorch.kernels.RBFKernel(active_dims=torch.tensor(active_dims_adjustment))

        if self.kernel_type == "rbf":
            self.adjustment_kernel = gpytorch.kernels.RBFKernel(active_dims=torch.tensor(active_dims_adjustment),
                                                            ard_num_dims=self.dim_adjustment)
        elif self.kernel_type == "matern":
            self.adjustment_kernel = gpytorch.kernels.MaternKernel(active_dims=torch.tensor(active_dims_adjustment),
                                                            ard_num_dims=self.dim_adjustment)
        elif self.kernel_type == "rq":
            self.adjustment_kernel = gpytorch.kernels.RQKernel(active_dims=torch.tensor(active_dims_adjustment),
                                                            ard_num_dims=self.dim_adjustment)
        else:
            raise ValueError("Kernel type not recognized.")

        self.mean_module = gpytorch.means.ConstantMean()
        # self.covar_module = gpytorch.kernels.ScaleKernel(
        #     self.treatment_kernel * self.condition_kernel * self.adjustment_kernel
        # )
        self.covar_module = self.treatment_kernel * self.condition_kernel * self.adjustment_kernel

    def forward(self, x):

        # check if the input has the right shape
        assert x.shape[1] == self.dim_treatment + self.dim_condition + self.dim_adjustment

        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)



