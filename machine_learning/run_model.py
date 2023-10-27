import torch.nn as nn
import torch

SAVE_PATH = './models'

# Model class must be defined somewhere
model = torch.load(SAVE_PATH)
model.eval()