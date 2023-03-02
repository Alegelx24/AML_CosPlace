import torch
import clip
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from torchvision.datasets import CIFAR10
from torchvision.transforms import *
from tqdm.notebook import tqdm


def load_models():
    device = 'cuda' if torch.cuda.is_available() else 'cpu' #useful???
    state_dicts = []
    
    #for f in os.listdir(directory_name_that_contains_different_best_model.pth)
    for f in os.listdir():
        print (f)
        if f[-2:] == 'pth':
            print(f'Loading {f}')
            state_dicts.append(torch.load(f, map_location=device))
    return state_dicts


def get_model_soup(model, state_dicts, alphal):
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  sd = {k : state_dicts[0][k].clone() * alphal[0] for k in state_dicts[0].keys()}
  for i in range(1, len(state_dicts)):
      for k in state_dicts[i].keys():
          sd[k] = sd[k] + state_dicts[i][k].clone() * alphal[i]
  model.load_state_dict(sd)
  model = model.to(device)
  return model

