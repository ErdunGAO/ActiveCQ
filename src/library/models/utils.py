
import torch

def expand_inverse_with_regularization(K_n_inv, k, k_nn, lambda_reg):
    """
    Compute the inverse of the extended matrix (K_{n+1} + lambda * I) 
    using the block matrix inversion formula.

    Parameters:
    - K_n_inv: torch.Tensor, shape (n, n), the inverse of (K_n + lambda * I_n).
    - k: torch.Tensor, shape (n, 1), similarity vector between the new data point and the existing n data points (as a column vector).
    - k_nn: torch.Tensor, shape (1, 1), self-similarity of the new data point (scalar).
    - lambda_reg: float, regularization term (lambda).

    Returns:
    - K_n1_inv: torch.Tensor, shape (n+1, n+1), the inverse of the extended matrix (K_{n+1} + lambda * I).
    """
    
    # Calculate mu = 1 / (k_nn + lambda_reg - k.T @ K_n_inv @ k)
    mu = 1.0 / (k_nn + lambda_reg - torch.matmul(k.T, torch.matmul(K_n_inv, k)))
    
    # Calculate the individual blocks of the matrix
    top_left = K_n_inv + mu * torch.matmul(torch.matmul(K_n_inv, k), torch.matmul(k.T, K_n_inv))
    top_right = -mu * torch.matmul(K_n_inv, k)
    bottom_left = -mu * torch.matmul(k.T, K_n_inv)
    bottom_right = mu

    # Combine the blocks into the (n+1) x (n+1) matrix
    K_n1_inv = torch.empty((K_n_inv.size(0) + 1, K_n_inv.size(1) + 1), dtype=K_n_inv.dtype)
    K_n1_inv[:K_n_inv.size(0), :K_n_inv.size(1)] = top_left
    K_n1_inv[:K_n_inv.size(0), -1] = top_right.squeeze()  # last column
    K_n1_inv[-1, :-1] = bottom_left.squeeze()  # last row
    K_n1_inv[-1, -1] = bottom_right  # bottom right

    return K_n1_inv