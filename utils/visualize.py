import matplotlib.pyplot as plt
import torch
import numpy as np

def plot_fiber_structure(hole_pos, hole_attr, global_attr=None, title=None, save_path=None):
    """
    可视化光纤结构：
        - 坐标点：空气孔位置
        - 半径：hole_attr[:, 0]
        - 颜色（可选）：hole_attr[:, 2:].argmax(dim=1)
    """
    if isinstance(hole_pos, torch.Tensor):
        hole_pos = hole_pos.cpu().numpy()
    if isinstance(hole_attr, torch.Tensor):
        hole_attr = hole_attr.cpu().numpy()

    radii = hole_attr[:, 0]
    if hole_attr.shape[1] > 2:
        colors = np.argmax(hole_attr[:, 2:], axis=1)
    else:
        colors = "blue"

    fig, ax = plt.subplots(figsize=(5, 5))
    for i in range(len(hole_pos)):
        circle = plt.Circle(hole_pos[i], radii[i], color='C{}'.format(colors[i] % 10) if isinstance(colors, np.ndarray) else colors,
                            alpha=0.7, edgecolor='black')
        ax.add_patch(circle)

    ax.set_aspect('equal')
    ax.autoscale_view()
    ax.set_xlabel("X (μm)")
    ax.set_ylabel("Y (μm)")
    ax.set_title(title or "Fiber Structure")

    if global_attr is not None:
        ax.text(0.01, 0.99, f"Global: {global_attr.cpu().numpy()}", transform=ax.transAxes,
                fontsize=8, verticalalignment='top')

    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()