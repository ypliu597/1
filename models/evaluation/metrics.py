import torch
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import pairwise_distances

def coverage_score(set_real, set_gen, threshold=0.1):
    """
    Coverage Score (COV): 生成结构中有多少真实结构被覆盖
    """
    dist_matrix = cdist(set_real, set_gen)
    min_distances = dist_matrix.min(axis=1)
    return np.mean(min_distances < threshold)

def matching_score(set_real, set_gen, k=1):
    """
    Matching Score (MMD): 真实结构与生成结构之间的平均最小距离
    """
    dist_matrix = cdist(set_real, set_gen)
    return np.mean(np.sort(dist_matrix, axis=1)[:, :k])

def wasserstein_distance_1D(a, b):
    """
    一维 Wasserstein 距离（适用于半径、坐标分布）
    """
    a_sorted = np.sort(a)
    b_sorted = np.sort(b)
    return np.mean(np.abs(a_sorted - b_sorted))

def radius_distribution_score(r_real, r_gen):
    """
    比较真实 vs 生成的半径分布差异（Wasserstein 距离）
    """
    return wasserstein_distance_1D(r_real, r_gen)

def coordinate_distribution_score(xy_real, xy_gen):
    """
    坐标点云结构的距离分布比较
    """
    d_real = pairwise_distances(xy_real)
    d_gen = pairwise_distances(xy_gen)
    return wasserstein_distance_1D(d_real.flatten(), d_gen.flatten())