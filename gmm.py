import math
import torch
import torch.nn.functional as F
import torch.nn as nn

def gmm_fitting(dino_features, y_hat, temp, return_adapter=False):
    K = y_hat.size(1)
    num_samples, d = dino_features.size()
    dino_features = dino_features.cuda()

    max_iter = 10
    std_init = 1 / d
    lambda_value = 1.0
    max_loss = 0.0

    y_hat = y_hat * 100
    y_hat, z = init_z(y_hat, softmax=True)

    mu = init_mu(K, d, z, dino_features)
    std = init_sigma(d, std_init)
    adapter = Gaussian(mu=mu, std=std).cuda()
    # W = build_affinity_matrix(dino_features, num_samples, n_neighbors)
    
    for k in range(max_iter + 1):
        gmm_likelihood = adapter(dino_features, no_exp=True)
        # z = update_z(gmm_likelihood, y_hat, z, W, lambda_value, n_neighbors)[0:num_samples]
        z, loss = update_z_wo_graph(gmm_likelihood, y_hat, z, lambda_value, temp)[0:num_samples]
        # z = update_z_wo_y_hat(gmm_likelihood, y_hat, z, lambda_value, temp)[0:num_samples]

        if k == max_iter:
            break

        adapter = update_mu(adapter, dino_features, z)
        adapter = update_sigma(adapter, dino_features, z)

    if return_adapter:
        return z, adapter
    else:
        return z


def update_z_wo_graph(gmm_likelihood, y_hat, z, lambda_value, temp=50.0):
    num_samples = gmm_likelihood.size(0)
    intermediate = gmm_likelihood.clone()
    intermediate -= torch.max(intermediate, dim=1, keepdim=True)[0]

    # compute loss1
    q = torch.exp(1 / 50 * intermediate)
    q = q / torch.sum(q, dim=1, keepdim=True)

    intermediate = y_hat * torch.exp(1 / temp * intermediate)
    z[0:num_samples] = intermediate / torch.sum(intermediate, dim=1, keepdim=True)
    
    # compute loss2
    p = y_hat
    kl_zp = F.kl_div(input=p.log(), target=z, reduction='batchmean')
    kl_zq = F.kl_div(input=q.log(), target=z, reduction='batchmean')
    loss = kl_zp + kl_zq  # loss = D_KL(z || p) + D_KL(z || q)
    # print(f' ***** kl_zp={kl_zp.item():.4f} ***** kl_zq={kl_zq.item():.4f} ***** loss={loss.item():.4f}')
    return z, loss


def update_z_wo_y_hat(gmm_likelihood, y_hat, z, lambda_value, temp=50.0):
    num_samples = gmm_likelihood.size(0)
    intermediate = gmm_likelihood.clone()
    intermediate -= torch.max(intermediate, dim=1, keepdim=True)[0]
    intermediate = torch.exp(1 / temp * intermediate)
    z[0:num_samples] = intermediate / torch.sum(intermediate, dim=1, keepdim=True)
    return z


class Gaussian(nn.Module):
    def __init__(self, mu, std):
        super().__init__()
        self.mu = mu.clone()
        self.K, self.num_components, self.d = self.mu.shape
        self.std = std.clone()
        self.mixing = torch.ones(self.K, self.num_components, device=self.mu.device) / self.num_components
        pass
    def forward(self, x, get_components=False, no_exp=False):
        chunk_size = 2500
        N = x.shape[0]
        M, D = self.mu.shape[0], self.std.shape[0]

        intermediate = torch.empty((N, M), dtype=x.dtype, device=x.device)

        for start_idx in range(0, N, chunk_size):
            end_idx = min(start_idx + chunk_size, N)
            # p_{i,k} ∝ exp(-½(f_i-μ_k)ᵀΣ⁻¹(f_i-μ_k)) 
            # p_{i,k} ∝ exp[-0.5 * (f_i-μ_k)² * 1/Σ)]
            # p_{i,k} = -(d/2) log(2π) exp[-0.5 * (f_i-μ_k)² * 1/Σ)]
            intermediate[start_idx:end_idx] = -0.5 * torch.einsum('ijk,ijk->ij', 
                                                                  (x[start_idx:end_idx][:, None, :] - self.mu[None, :, 0, :]) ** 2, 
                                                                  1 / self.std[None, None, :])
        if not no_exp:
            intermediate = torch.exp(intermediate)

        if get_components:
            return torch.ones_like(intermediate.unsqueeze(1))
        return intermediate

    def set_std(self, std):
        self.std = std


def update_z(gmm_likelihood, y_hat, z, W, lambda_value, n_neighbors, max_iter=5):
    num_samples = gmm_likelihood.size(0)
    for it in range(max_iter):
        intermediate = gmm_likelihood.clone()
        intermediate += (50 / (n_neighbors * 2)) * (W.T @ z + (W @ z[0:num_samples, :])[0:num_samples, :])
        intermediate -= torch.max(intermediate, dim=1, keepdim=True)[0] # For numerical stability
        intermediate = (y_hat ** lambda_value) * torch.exp(1 / 50 * intermediate)
        z[0:num_samples] = intermediate / torch.sum(intermediate, dim=1, keepdim=True)
    return z


def update_mu(adapter, query_features, z):
    affinity_unlabeled = z
    n_query = affinity_unlabeled.size(0)
    weights = (1 / n_query) * affinity_unlabeled
    # Use einsum to compute the new_mu for each class in one pass
    new_mu = torch.einsum('ij,ik->jk', weights, query_features)
    new_mu /= (1 / n_query * torch.sum(affinity_unlabeled, dim=0).unsqueeze(-1))
    new_mu = new_mu.unsqueeze(1)
    new_mu /= new_mu.norm(dim=-1, keepdim=True)
    adapter.mu = new_mu
    return adapter


def update_sigma(adapter, query_features, z):
    affinity_unlabeled = z
    n_query = affinity_unlabeled.size(0)
    std = 0

    chunk_size = 2500  # Iterate over query_features in chunks to avoid large memory consumption

    for start_idx in range(0, n_query, chunk_size):
        end_idx = min(start_idx + chunk_size, n_query)
        query_features_chunk = query_features[start_idx:end_idx]

        # Compute the weighted sum of squared differences for the chunk
        chunk_result = (1 / n_query) * torch.einsum('ij,ijk->k', affinity_unlabeled[start_idx:end_idx, :],
                                                    (query_features_chunk[:, None, :] - adapter.mu[None, :, 0, :]) ** 2)

        # If this is the first chunk, initialize std; otherwise, accumulate
        if start_idx == 0:
            std = chunk_result
        else:
            std += chunk_result

    std /= (1 / n_query * torch.sum(affinity_unlabeled[:,:]))
    adapter.set_std(std)
    return adapter


def init_z(affinity, softmax=True):
    if softmax:
        y_hat = F.softmax(affinity, dim=1)
        z = F.softmax(affinity, dim=1)
    else:
        y_hat = affinity
        z = affinity
    return y_hat, z


def init_mu(K, d, z, query_features):
    mu = torch.zeros(K, 1, d, device=query_features.device)
    n_most_confident = 8
    topk_values, topk_indices = torch.topk(z, k=n_most_confident, dim=0)  # 8 pseudo-labels per class

    mask = torch.zeros_like(z).scatter_(0, topk_indices, 1)
    filtered_z = z * mask
    for c in range(K):
        class_indices = mask[:, c].nonzero().squeeze(1)
        class_features = query_features[class_indices]
        class_z = filtered_z[class_indices, c].unsqueeze(1)

        combined = class_features * class_z
        component_mean = combined[:n_most_confident].mean(dim=0)
        mu[c, 0, :] = component_mean
    mu /= mu.norm(dim=-1, keepdim=True)
    return mu


def init_sigma(d, std_init):
    std = (torch.eye(d).diag() * std_init).cuda()
    return std


def build_affinity_matrix(query_features, num_samples, n_neighbors=3):
    device = query_features.device
    affinity = query_features.matmul(query_features.T).cpu()
    num_rows = num_samples
    num_cols = num_samples
    knn_index = affinity.topk(n_neighbors + 1, -1, largest=True).indices[:, 1:]
    row_indices = torch.arange(num_rows).unsqueeze(1).repeat(1, n_neighbors).flatten()
    col_indices = knn_index.flatten()
    values = affinity[row_indices, col_indices].to(device)
    W = torch.sparse_coo_tensor(torch.stack([row_indices, col_indices]).to(device), values, size=(num_rows, num_cols), device=device)
    return W
