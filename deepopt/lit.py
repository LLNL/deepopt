from typing import Union

import lightning as L
import torch
import torch.nn as nn
from torch.optim import SGD, Adam

from deepopt.surrogate_utils import MLP as Arch


class LitDelUQ(L.LightningModule):
    """
    Pytorch Lightning class to encapsulate neural networks used for Delta-UQ
    """
    
    def __init__(
        self,
        network: Arch,
        optimizer: Union[Adam, SGD],
        target: str = "dy"):
        super().__init__()
        self.network = network
        self.optimizer = optimizer
        self.target = target
        
    def training_step(self,batch,batch_idx):
        x,y = batch
        x = self.network.input_mapping(x)
        flipped_x = torch.flip(x,[0])
        diff_x = x - flipped_x
        inp = torch.cat([flipped_x,diff_x],axis=1)
        
        if self.target == "y":
            out = y
        else:
            flipped_y = torch.flip(y,[0])
            diff_y = y - flipped_y
            out = diff_y
            
        out_hat = self.network(inp)
        loss = nn.functional.mse_loss(out_hat.float(), out.float())
        return loss
        
        
    def configure_optimizers(self):
        return self.optimizer