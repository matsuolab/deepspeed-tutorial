import os
import yaml
import json
import torch
import random
import numpy as np

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)