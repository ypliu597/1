import os
import torch
import matplotlib.pyplot as plt
from pytorch_lightning import Callback
from core.utils.visualize import plot_fiber_structure

class VisualizeSampleCallback(Callback):
    def __init__(self, sample_dir="images", sample_interval=5):
        super().__init__()
        self.sample_dir = sample_dir
        self.sample_interval = sample_interval
        os.makedirs(self.sample_dir, exist_ok=True)

        # 示例目标参数（可替换为动态或预设）
        self.sample_target_param = torch.tensor([0.7, 1.3, 0.08, -25.0, 35.0, 1.55])

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.sample_interval == 0:
            self.sample_and_save(trainer, pl_module)

    def sample_and_save(self, trainer, pl_module):
        model = pl_module.model  # BFN4FiberDesign
        device = next(model.parameters()).device
        model.eval()

        with torch.no_grad():
            coords, attr, global_attr = model.sample(self.sample_target_param.unsqueeze(0).to(device))

        fig, ax = plt.subplots(figsize=(5, 5))
        plot_fiber_structure(coords.cpu(), attr.cpu(), global_attr.cpu(),
                             title=f"Epoch {trainer.current_epoch}", save_path=None)
        fig.savefig(os.path.join(self.sample_dir, f"epoch_{trainer.current_epoch}.png"))
        plt.close(fig)