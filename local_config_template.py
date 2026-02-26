import torch
from enum import Enum
import os

class DiffusionModelType(Enum):
    DDIM = 'ddim'
    CM = 'cm'

class RunType(Enum):
    PROMPTS = "PROPMTS"
    SEGMENTATIONS = "SEGMENTATIONS"
    BOTH = "BOTH"
    

device = ""
d_type = torch.float16
batch_size = 1
limit_no_of_batches = 30
coco_path = ''
voc_path = ''
context_path = ''

# diffusion_timestamps
diffusion_timestamps = [[0.1]]
diffusion_model = DiffusionModelType.DDIM

run_type = RunType.SEGMENTATIONS

prompts_dir = './prompts'
prompts_path = os.path.join(prompts_dir, 'prompts.json')