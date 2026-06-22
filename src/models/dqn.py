import torch
import torch.nn as nn
import torch.nn.functional as F
from src import config_defaults as hyp

from pickletools import optimize
import rdkit
import rdkit.Chem.Descriptors as des
from rdkit import Chem
import numpy as np
import pandas as pd
from sklearn import preprocessing

import time


# import pytorch_lightning as pl
# from resnet import resnet50, resnet10, resnet18, resnet200


# class LitModel(pl.LightningModule):
#   def __init__(self, num_layers=50):
#     super().__init__()
#     if num_layers == 50:
#       model = resnet50
#     elif num_layers == 10:
#       model = resnet10
#     elif num_layers == 18:
#       model = resnet18
#     elif num_layers == 200:
#       model = resnet200
#     self.model = model(num_classes=1,
#                        sample_size=32,
#                        sample_duration=32,
#                        num_channels=1)
#   def forward(self, x):
#     return self.model(x)
  
#   def training_step(self, batch, batch_idx):
#     grid, y = batch
#     y_hat = self(grid)
#     loss = F.mse_loss(y_hat, y.unsqueeze(1).float())
#     self.log("train/loss", loss)
#     return loss
  
#   def validation_step(self, val_batch, batch_idx):
#     grid, y = val_batch
#     y_hat = self(grid)
#     torch.save(y, f"val_real_batch_{batch_idx}.pt")
#     torch.save(y_hat, f"val_pred_batch_{batch_idx}.pt")
#     loss = F.mse_loss(y_hat, y.unsqueeze(1).float())

#     self.log('val_loss', loss)
#     self.log("validation/loss", loss)
#     return loss

#   def test_step(self, test_batch, batch_idx):
#     grid, y = test_batch
#     y_hat = self(grid)
#     loss = F.mse_loss(y_hat, y.unsqueeze(1).float())
#     torch.save(y, f"test_real_batch_{batch_idx}.pt")
#     torch.save(y_hat, f"test_pred_batch_{batch_idx}.pt")
#     self.log("test_loss/loss", loss)
#     return loss

#   def configure_optimizers(self):
#     optimizer = torch.optim.RMSprop(self.parameters(), lr=1e-3)
#     lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
#     return [optimizer], [lr_scheduler]
#     # return optimizer



class MolDQN(nn.Module):
    def __init__(self, input_length, output_length):
        super(MolDQN, self).__init__()

        self.linear_1 = nn.Linear(input_length, 1024)
        self.linear_2 = nn.Linear(1024, 512)
        self.linear_3 = nn.Linear(512, 128)
        self.linear_4 = nn.Linear(128, 32)
        self.linear_5 = nn.Linear(32, output_length)

        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.linear_1(x))
        x = self.activation(self.linear_2(x))
        x = self.activation(self.linear_3(x))
        x = self.activation(self.linear_4(x))
        x = self.linear_5(x)

        return x

    def training_step(self, states, next_states, rewards, dones, batch_size):
        pass

    # def forward_observation(self, observation):



# class MolDQN(nn.Module):
#     def __init__(self, input_length, output_length, width_factor = 1):
#         super(MolDQN, self).__init__()

#         self.linear_1 = nn.Linear(input_length, 1024*width_factor)
#         self.linear_2 = nn.Linear(1024*width_factor, 512*width_factor)
#         self.linear_3 = nn.Linear(512*width_factor, 128*width_factor)
#         self.linear_4 = nn.Linear(128*width_factor, 32*width_factor)
#         self.linear_5 = nn.Linear(32*width_factor, output_length)

#         self.activation = nn.ReLU()

#     def forward(self, x):
#         x = self.activation(self.linear_1(x))
#         x = self.activation(self.linear_2(x))
#         x = self.activation(self.linear_3(x))
#         x = self.activation(self.linear_4(x))
#         x = self.linear_5(x)

#         return x

width_factor = 1
class MolDQN_2(nn.Module):
    def __init__(self, input_length, output_length):
        super(MolDQN_2, self).__init__()

        self.linear_1 = nn.Linear(input_length, 1024*width_factor)
        self.linear_2 = nn.Linear(1024*width_factor, 512*width_factor)
        self.linear_3 = nn.Linear(512*width_factor, 128*width_factor)
        self.linear_4 = nn.Linear(128*width_factor, 32*width_factor)
        self.linear_5 = nn.Linear(32*width_factor, output_length)

        self.activation = nn.ReLU()

    def forward(self, x, step_left):
        x = self.activation(self.linear_1(x))
        x = self.activation(self.linear_2(x))
        x = self.activation(self.linear_3(x))
        x = self.activation(self.linear_4(x))
        x = self.linear_5(x)

        return x * hyp.discount_factor ** step_left
