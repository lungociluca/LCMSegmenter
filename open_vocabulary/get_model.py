import os
import sys
sys.path.insert(0, os.path.join(sys.path[0], ".."))

from diffusers import StableDiffusionPipeline,  DDIMScheduler, DiffusionPipeline
from local_config import DiffusionModelType, diffusion_model, device, d_type

local_files_only = False

def get_diffusion_model():
    if diffusion_model == DiffusionModelType.DDIM:
        model_key = "runwayml/stable-diffusion-v1-5"
        ldm_stable = StableDiffusionPipeline.from_pretrained(model_key, local_files_only=local_files_only,torch_dtype=d_type).to(device)
        ldm_stable.scheduler = DDIMScheduler.from_pretrained(model_key, local_files_only=local_files_only, subfolder="scheduler")
        diffusion_timestamps = 1000

    elif diffusion_model == DiffusionModelType.CM:
        model_key = "SimianLuo/LCM_Dreamshaper_v7"
        ldm_stable = DiffusionPipeline.from_pretrained(model_key, local_files_only=local_files_only, torch_dtype=d_type)
        ldm_stable.to(device)
        ldm_stable.to(torch_device=device, torch_dtype=d_type)
        diffusion_timestamps = 50
        
    else:
        raise TypeError("Could not find a suitable diffusion model type in local_config.py")

    return ldm_stable, diffusion_timestamps


if __name__ == '__main__':
    model, _ = get_diffusion_model()
    components = [
        ('Vae', model.vae), 
        ('Unet', model.unet), 
        # ('Tokenizer', model.tokenizer)
    ]
    total_params = 0
    for name, comp in components:
        params_count = sum(p.numel() for p in comp.parameters())
        print(f'{name}: {params_count}')
        total_params += params_count
    print("Total count", total_params)