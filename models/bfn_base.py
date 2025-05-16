import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean  # SBDD 版本使用了 scatter_sum，但您之前的版本用了 scatter_mean，这里保留 scatter_mean
import torch.distributions as dist  # 需要从 SBDD 版本引入

import numpy as np
import warnings  # 用于打印警告而不是直接print

LOG2PI = np.log(2 * np.pi)


class BFNBase(nn.Module):
    def __init__(self, *args, **kwargs):  # 保持与 SBDD 一致的构造函数签名
        super(BFNBase, self).__init__(*args, **kwargs)

    def continuous_var_bayesian_update(self, t, sigma1, x):
        """
        连续变量的贝叶斯更新（如坐标）
        Eq: θ ~ N(μ=γ(t)x, σ²=γ(t)(1−γ(t)))
        x: [N, D]
        sigma1: scalar tensor
        t: [N, 1] or scalar tensor broadcastable
        """
        # gamma = 1 - torch.pow(sigma1, 2 * t) # Original fiber version
        # SBDD version uses element-wise power, which is fine if sigma1 is scalar
        # Ensuring t has the same device as sigma1 if it's a tensor
        if isinstance(t, torch.Tensor) and t.device != sigma1.device:
            t = t.to(sigma1.device)

        gamma = 1 - torch.pow(sigma1.to(x.device), 2 * t.to(x.device))  # [N,1] or [1]

        if torch.isnan(gamma).any():
            warnings.warn("NaN in gamma (coord) during continuous_var_bayesian_update.", RuntimeWarning)
            # Handle NaN gamma, e.g., by setting it to a safe value or re-raising
            gamma = torch.where(torch.isnan(gamma), torch.zeros_like(gamma), gamma)

        # Variance term: gamma * (1 - gamma) can be negative if gamma > 1 or gamma < 0 due to numerical issues
        # Clamp gamma to [eps, 1-eps] to ensure variance is non-negative
        eps_gamma = 1e-8
        clamped_gamma = torch.clamp(gamma, eps_gamma, 1.0 - eps_gamma)
        variance = clamped_gamma * (1 - clamped_gamma)

        # Ensure variance is non-negative before sqrt
        mu = gamma * x + torch.randn_like(x) * torch.sqrt(
            torch.relu(variance) + 1e-10)  # Added relu and slightly larger eps for sqrt
        return mu, gamma

    def discrete_var_bayesian_update(self, t, beta1, x, K):
        """
        离散变量的贝叶斯更新（如孔属性、材料的One-Hot编码）
        x: [N, K] (typically one-hot)
        beta1: scalar tensor
        t: [N, 1] or scalar tensor broadcastable
        K: int, number of classes
        """
        # Ensuring t has the same device as beta1 if it's a tensor
        if isinstance(t, torch.Tensor) and t.device != beta1.device:
            t = t.to(beta1.device)

        beta = beta1.to(x.device) * (t.to(x.device) ** 2)  # [N,1] or [1]

        one_hot_x = x
        mean_logits = beta * (K * one_hot_x - 1)
        # Ensure std is calculated safely, beta and K are positive
        std_logits = (torch.abs(beta) * K).sqrt()  # abs(beta) for safety, though beta should be positive

        if torch.isnan(mean_logits).any() or torch.isnan(std_logits).any():
            warnings.warn("NaN in mean_logits/std_logits (discrete_var_bayesian_update).", RuntimeWarning)
            # Handle NaN, e.g., by setting to zero or re-raising
            mean_logits = torch.where(torch.isnan(mean_logits), torch.zeros_like(mean_logits), mean_logits)
            std_logits = torch.where(torch.isnan(std_logits), torch.ones_like(std_logits), std_logits)

        eps = torch.randn_like(mean_logits)
        y_logits = mean_logits + std_logits * eps
        y_logits = y_logits.clamp(-10, 10)  # 避免 softmax 数值爆炸 (from your original)
        theta_probs = F.softmax(y_logits, dim=-1)
        return theta_probs

    def ctime4continuous_loss(self, t, sigma1, x_pred, x, segment_ids=None):
        """
        连续时间下的坐标 KL loss（与 sender 的加权平方差）
        Based on SBDD version with segment_ids, adapted from your original if segment_ids is None.
        t: [N_nodes_in_batch, 1] or broadcastable
        sigma1: scalar
        x_pred: [N_nodes_in_batch, D_coord]
        x: [N_nodes_in_batch, D_coord] (ground truth)
        segment_ids: [N_nodes_in_batch] (batch index for each node)
        """
        # Ensure t is on the same device as sigma1
        if isinstance(t, torch.Tensor) and t.device != sigma1.device:
            t = t.to(sigma1.device)
        sigma1 = sigma1.to(x.device)

        # Weight term: (sigma1)^(-2t)
        # Ensure t is correctly shaped for broadcasting with weight
        t_reshaped = t.view(-1)  # Flatten t to [N_nodes_in_batch] for weight calculation
        if t_reshaped.shape[0] != x.shape[0]:  # If t was scalar, expand it
            t_reshaped = t_reshaped.expand(x.shape[0])

        weight = torch.pow(sigma1, -2 * t_reshaped)  # [N_nodes_in_batch]

        # Squared error term, summed over feature dimensions
        squared_error = ((x_pred - x).view(x.shape[0], -1).abs().pow(2)).sum(dim=1)  # [N_nodes_in_batch]

        loss_per_node = weight * squared_error  # [N_nodes_in_batch]

        if segment_ids is not None:
            # Average loss per graph/segment
            loss = scatter_mean(loss_per_node, segment_ids, dim=0)  # [Num_graphs]
        else:
            # If no segments, just take the mean over all nodes
            loss = loss_per_node  # Still [N_nodes_in_batch], will be averaged later by .mean() call

        # The final factor -torch.log(sigma1) is applied after averaging if segment_ids is None
        # or to each graph's mean loss if segment_ids is not None.
        # If loss is already [Num_graphs], apply element-wise.
        # If loss is [N_nodes_in_batch], it means no segmentation, so apply to all then mean later.
        final_loss = -torch.log(sigma1) * loss
        return final_loss  # Shape: [Num_graphs] or [N_nodes_in_batch]

    def dtime4continuous_loss(self, i, N, sigma1, x_pred, x, segment_ids=None):
        """
        离散时间下的坐标 KL loss（采样步 i）
        i: tensor of current step indices [N_nodes_in_batch, 1] or broadcastable
        N: int, total discrete steps
        sigma1: scalar
        x_pred: [N_nodes_in_batch, D_coord]
        x: [N_nodes_in_batch, D_coord] (ground truth)
        segment_ids: [N_nodes_in_batch] (batch index for each node)
        """
        # Ensure i is on the same device as sigma1
        if isinstance(i, torch.Tensor) and i.device != sigma1.device:
            i = i.to(sigma1.device)
        sigma1 = sigma1.to(x.device)

        # Denominator stability from your original version
        denominator = 2 * torch.pow(sigma1, 2 * i / N) + 1e-8  # Ensure i/N is float division
        weight_numerator = N * (1 - torch.pow(sigma1, 2 / N))

        weight = weight_numerator / denominator  # Shape will depend on i

        # Ensure weight is correctly shaped for broadcasting, t_reshaped from ctime might be a good model
        if weight.ndim > 0 and weight.shape[0] == 1 and x.shape[0] > 1:  # If weight is scalar-like but x is not
            weight = weight.expand(x.shape[0])
        elif weight.ndim > 0 and weight.shape[0] != x.shape[0] and i.shape[0] == x.shape[0]:  # if i was [N_nodes,1]
            weight = weight.view(-1)

        squared_error = ((x_pred - x).view(x.shape[0], -1).abs().pow(2)).sum(dim=-1)  # [N_nodes_in_batch]
        loss_per_node = weight * squared_error  # [N_nodes_in_batch]

        if segment_ids is not None:
            loss = scatter_mean(loss_per_node, segment_ids, dim=0)  # [Num_graphs]
        else:
            loss = loss_per_node  # [N_nodes_in_batch], will be averaged later

        return loss  # Shape: [Num_graphs] or [N_nodes_in_batch]

    # --- NEW DISCRETE LOSS FUNCTIONS FROM SBDD BFNBase ---
    def ctime4discrete_loss(self, t, beta1, one_hot_x, p_0, K, segment_ids=None):
        # Eq.(205): L∞(x) = Kβ(1) E_{t∼U (0,1), p_F (θ|x,t)} [t|e_x − e_hat(θ, t)|**2,
        # where e_hat(θ, t) = (\sum_k p_O^(1) (k | θ; t)e_k, ..., \sum_k p_O^(D) (k | θ; t)e_k)
        # t: [N_nodes_in_batch, 1] or broadcastable
        # beta1: scalar
        # one_hot_x: [N_nodes_in_batch, K] (true one-hot labels)
        # p_0: [N_nodes_in_batch, K] (predicted probabilities from model)
        # K: int, number of classes
        # segment_ids: [N_nodes_in_batch]

        # Ensure t is on the same device as beta1
        if isinstance(t, torch.Tensor) and t.device != beta1.device:
            t = t.to(beta1.device)
        beta1 = beta1.to(one_hot_x.device)

        e_x = one_hot_x
        e_hat = p_0
        assert e_x.size() == e_hat.size(), f"Shape mismatch: e_x {e_x.shape}, e_hat {e_hat.shape}"

        # Ensure t has a compatible shape for element-wise multiplication, typically [N_nodes_in_batch]
        t_reshaped = t.view(-1)
        if t_reshaped.shape[0] != e_x.shape[0]:
            if t_reshaped.shape[0] == 1:  # Scalar t
                t_reshaped = t_reshaped.expand(e_x.shape[0])
            else:
                # This case might indicate an issue with how t is passed or reshaped
                # For example, if t was originally for graphs, and now needed for nodes
                raise ValueError(f"Shape mismatch for t_reshaped {t_reshaped.shape} and e_x {e_x.shape}")

        squared_error_sum = ((e_x - e_hat) ** 2).sum(dim=-1)  # [N_nodes_in_batch]
        loss_per_node = K * beta1 * t_reshaped * squared_error_sum  # [N_nodes_in_batch]

        if segment_ids is not None:
            L_infinity = scatter_mean(loss_per_node, segment_ids, dim=0)  # [Num_graphs]
        else:
            L_infinity = loss_per_node  # [N_nodes_in_batch]

        return L_infinity  # Shape: [Num_graphs] or [N_nodes_in_batch]

    def dtime4discrete_loss_prob(
            self, i, N, beta1, one_hot_x, p_0, K, n_samples=200, segment_ids=None
    ):
        # Based on the official BFN implementation (sampling-based loss)
        # i: tensor of current step indices [N_nodes_in_batch, 1] or broadcastable
        # N: int, total discrete steps
        # beta1: scalar
        # one_hot_x: [N_nodes_in_batch, K] (true one-hot labels)
        # p_0: [N_nodes_in_batch, K] (predicted probabilities from model)
        # K: int, number of classes
        # segment_ids: [N_nodes_in_batch]

        # Ensure i is on the same device as beta1
        if isinstance(i, torch.Tensor) and i.device != beta1.device:
            i = i.to(beta1.device)
        beta1 = beta1.to(one_hot_x.device)

        target_x = one_hot_x  # Shape: [num_elements, K] (num_elements could be N_nodes_in_batch)
        pred_probs = p_0  # Shape: [num_elements, K]

        # alpha for this discrete time step. Ensure i/N is float division.
        alpha_val = beta1 * (2 * i.float() - 1) / (N ** 2)  # Shape: [num_elements, 1] or broadcastable

        # Ensure alpha_val has shape [num_elements, 1] for broadcasting
        if alpha_val.ndim == 0 or alpha_val.shape[0] == 1 and target_x.shape[0] > 1:
            alpha_val = alpha_val.expand(target_x.shape[0], 1)
        elif alpha_val.shape[0] != target_x.shape[0]:
            raise ValueError(f"Shape mismatch for alpha_val {alpha_val.shape} and target_x {target_x.shape}")
        alpha_val = alpha_val.view(-1, 1)  # Ensure [num_elements, 1]

        # Sender distribution: N(y | alpha * (K * target_x - 1), alpha * K * I)
        # target_x is one-hot
        mean_sender = alpha_val * (K * target_x - 1)  # [num_elements, K]
        std_sender = (K * torch.abs(alpha_val)) ** 0.5  # [num_elements, 1], abs for safety
        sender_dist = dist.Independent(dist.Normal(mean_sender, std_sender), 1)

        # Receiver distribution (Mixture of Gaussians)
        # For each of K possible original classes e_k:
        # component_k_mean = alpha * (K * e_k - 1)
        # component_k_std = (K * alpha)**0.5
        # receiver_components: dist.Normal, with batch_shape [num_elements, K] (one normal per original class e_k)
        # event_shape [K] (dimension of y)

        # Create e_k: standard basis vectors (one-hot encodings for each class)
        # e_k_all_classes will be [K, K]
        e_k_all_classes = torch.eye(K, device=target_x.device, dtype=target_x.dtype)

        # Expand alpha_val and e_k_all_classes for broadcasting
        # alpha_val: [num_elements, 1, 1]
        # e_k_all_classes: [1, K, K]
        expanded_alpha = alpha_val.unsqueeze(-1)
        expanded_e_k = e_k_all_classes.unsqueeze(0)

        mean_receiver_components = expanded_alpha * (K * expanded_e_k - 1)  # [num_elements, K, K]
        std_receiver_components = (K * torch.abs(expanded_alpha)) ** 0.5  # [num_elements, 1, 1]

        # Normal distribution for each component of the mixture
        # Batch shape [num_elements, K], event shape [K]
        receiver_components = dist.Independent(
            dist.Normal(mean_receiver_components, std_receiver_components), 1
        )

        # Mixture distribution
        # pred_probs are the mixture weights, shape [num_elements, K]
        receiver_mix_distribution = dist.Categorical(probs=pred_probs)
        receiver_dist = dist.MixtureSameFamily(receiver_mix_distribution, receiver_components)

        # Monte Carlo estimate of KL divergence (or related quantity)
        y_samples = sender_dist.sample(torch.Size([n_samples]))  # [n_samples, num_elements, K]

        log_prob_sender = sender_dist.log_prob(y_samples)  # [n_samples, num_elements]
        log_prob_receiver = receiver_dist.log_prob(y_samples)  # [n_samples, num_elements]

        # The loss is N * (E_y[log q(y|x_0)] - E_y[log p(y|x_hat_0)])
        # This is related to -N * ELBO or +N * KL divergence
        loss_per_element = N * (log_prob_sender - log_prob_receiver).mean(dim=0)  # [num_elements]

        if segment_ids is not None:
            loss = scatter_mean(loss_per_element, segment_ids, dim=0)  # [Num_graphs]
        else:
            loss = loss_per_element  # [num_elements]

        return loss.mean()  # Return mean loss over batch/elements if not segmented, or mean over graph means

    def dtime4discrete_loss(self, i, N, beta1, one_hot_x, p_0, K, segment_ids=None):
        # Based on Algorithm 7 in BFN paper (non-sampling based discrete loss for discrete time)
        # i: tensor of current step indices [N_nodes_in_batch, 1] or broadcastable
        # N: int, total discrete steps
        # beta1: scalar
        # one_hot_x: [N_nodes_in_batch, K] (true one-hot labels)
        # p_0: [N_nodes_in_batch, K] (predicted probabilities from model)
        # K: int, number of classes
        # segment_ids: [N_nodes_in_batch]

        if isinstance(i, torch.Tensor) and i.device != beta1.device:
            i = i.to(beta1.device)
        beta1 = beta1.to(one_hot_x.device)

        e_x_true = one_hot_x  # True one-hot, shape [num_elements, K]
        e_hat_probs = p_0  # Predicted probabilities, shape [num_elements, K]
        assert e_x_true.size() == e_hat_probs.size()

        alpha_val = beta1 * (2 * i.float() - 1) / (N ** 2)  # Shape [num_elements, 1] or broadcastable
        if alpha_val.ndim == 0 or alpha_val.shape[0] == 1 and e_x_true.shape[0] > 1:
            alpha_val = alpha_val.expand(e_x_true.shape[0], 1)
        elif alpha_val.shape[0] != e_x_true.shape[0]:
            raise ValueError(f"Shape mismatch for alpha_val {alpha_val.shape} and e_x_true {e_x_true.shape}")
        alpha_val = alpha_val.view(-1, 1)  # Ensure [num_elements, 1]

        # Calculate y_ sampled from q(y | x_0_true, t)
        # mean_true_logits = alpha_val * (K * e_x_true - 1)    # [num_elements, K]
        # std_true_logits = (K * torch.abs(alpha_val))**0.5     # [num_elements, 1]
        # y_sampled = mean_true_logits + std_true_logits * torch.randn_like(mean_true_logits) # [num_elements, K]

        # Simpler: use the expected structure from the paper for L_N
        # L_N = -log sum_k' p_0(k') q(y_sampled | k', t)
        # where y_sampled is drawn based on x_true.

        # Paper's Algo 7 step 4: Sample y ~ N(y | α(K e_x - 1), α K I)
        mean_y_ = alpha_val * (K * e_x_true - 1)
        std_y_ = (K * torch.abs(alpha_val)) ** 0.5
        y_ = mean_y_ + std_y_ * torch.randn_like(mean_y_)  # y_ shape [num_elements, K]

        # Paper's Algo 7 step 5: Compute log p(y | e_k', α) for all k'
        # p(y | e_k', α) = N(y | α(K e_k' - 1), α K I)

        # e_k_prime_all_classes will be [K, K] (standard basis vectors)
        e_k_prime_all_classes = torch.eye(K, device=e_x_true.device, dtype=e_x_true.dtype)

        # Expand for broadcasting:
        # y_: [num_elements, 1, K]
        # alpha_val: [num_elements, 1, 1]
        # e_k_prime_all_classes: [1, K, K]
        y_expanded = y_.unsqueeze(1)
        alpha_expanded = alpha_val.unsqueeze(-1)

        mean_for_likelihood = alpha_expanded * (K * e_k_prime_all_classes.unsqueeze(0) - 1)  # [num_elements, K, K]
        std_for_likelihood = (K * torch.abs(alpha_expanded)) ** 0.5  # [num_elements, 1, 1]

        # Log Normal PDF: log N(y | mu, sigma^2) = -0.5 * log(2pi*sigma^2) - 0.5 * ((y-mu)/sigma)^2
        # log_q_y_given_k_prime is log p(y_sampled | e_k', alpha)
        log_q_y_given_k_prime = (
                -0.5 * LOG2PI - torch.log(std_for_likelihood)  # Log std part
                - 0.5 * ((y_expanded - mean_for_likelihood) / std_for_likelihood) ** 2  # Exponent part
        ).sum(dim=-1)  # Sum over the K dimensions of y, result shape [num_elements, K_classes_for_k_prime]

        # Paper's Algo 7 step 6: L_N = -log sum_k' p_0(k') exp(log p(y | e_k', α))
        # logsumexp( log p_0(k') + log p(y | e_k', α) )
        if torch.isinf(log_q_y_given_k_prime).any() or torch.isnan(log_q_y_given_k_prime).any():
            warnings.warn("Non-finite values in log_q_y_given_k_prime in dtime4discrete_loss", RuntimeWarning)
            log_q_y_given_k_prime = torch.nan_to_num(log_q_y_given_k_prime, nan=0.0, posinf=0.0, neginf=-1e8)

        # Add log probs: log(e_hat_probs) + log_q_y_given_k_prime
        # Clamp e_hat_probs to avoid log(0)
        log_likelihood_terms = torch.log(
            torch.clamp(e_hat_probs, min=1e-20)) + log_q_y_given_k_prime  # [num_elements, K]

        log_sum_exp_likelihood = torch.logsumexp(log_likelihood_terms, dim=-1)  # [num_elements]

        L_N_per_element = -log_sum_exp_likelihood  # [num_elements]

        if segment_ids is not None:
            L_N = scatter_mean(L_N_per_element, segment_ids, dim=0)  # [Num_graphs]
        else:
            L_N = L_N_per_element  # [num_elements]

        return N * L_N.mean()  # Return final mean loss

    # --- Abstract methods to be implemented by subclasses ---
    def interdependency_modeling(self, *args, **kwargs):  # Matched SBDD signature style
        raise NotImplementedError("You must implement the structure decoder Φ here.")

    def forward(self, *args, **kwargs):  # Matched SBDD signature style
        raise NotImplementedError

    def loss_one_step(self, *args, **kwargs):  # Matched SBDD signature style
        raise NotImplementedError

    def sample(self, *args, **kwargs):  # Matched SBDD signature style
        raise NotImplementedError