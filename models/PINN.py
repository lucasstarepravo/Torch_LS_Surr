from torch import Tensor
from models.NN_Base import BaseModel
import torch


def calc_moments_torch(stand_f, pred_l, n):
    """ Calculate right-hand side vector of the linear system with the predicted solution vector """
    if stand_f.ndim != 3:
        raise ValueError("stand_f must have 3 dimensions after reshaping.")
    stand_f = stand_f.view(stand_f.shape[0], n, n)
    pred_l = pred_l.unsqueeze(-1)
    moments = torch.matmul(stand_f, pred_l)
    return moments.squeeze(-1)


class PINN(BaseModel):
    def __init__(self,
                 alpha: float,
                 moments_order: int,
                 hidden_layers: list,
                 optimizer: str,
                 loss_function: str,
                 epochs: int,
                 batch_size: int,
                 train_f: Tensor,
                 train_l: Tensor) -> None:

        super().__init__(hidden_layers, optimizer, loss_function, epochs, batch_size, train_f, train_l)
        self.alpha = alpha
        self.moments_order = int(moments_order)

    def physics_loss_fn(self, outputs, inputs):
        n = int((self.moments_order ** 2 + 3 * self.moments_order) / 2)
        moments = calc_moments_torch(inputs, outputs, n)
        target_moments = torch.zeros((outputs.shape[0], n), device=outputs.device)
        target_moments[:, 2] = 1
        target_moments[:, 4] = 1
        physics_loss = (target_moments - moments) ** 2
        return physics_loss.mean(axis=0)

    def calculate_loss(self, outputs, labels, inputs=None):
        if inputs is None:
            raise ValueError("Inputs cannot be None for this child class.")
        physics_loss = self.physics_loss_fn(outputs, inputs)
        data_loss = self.loss_function(outputs, labels)
        return (1 - self.alpha) * data_loss + self.alpha * physics_loss.mean()


    def save_checkpoint(self, path_to_save, model_type, model_ID, model_ddp, **kwargs):
        extra_attrs = {'alpha': self.alpha,
                       'moments_order': self.moments_order}
        super().save_checkpoint(path_to_save, model_type, model_ID, model_ddp, **extra_attrs)
