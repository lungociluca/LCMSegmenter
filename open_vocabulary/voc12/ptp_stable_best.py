from typing import Optional, Union, Tuple, List, Callable, Dict
import sys
# sys.path.append("./../..")
sys.path.insert(0, sys.path[0]+"/../..")
import scipy.io as sio
import torch
import numpy as np
import abc
import ptp_utils
import visual_code.seq_aligner as seq_aligner
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from dataset.mydatasets import build_dataset
import argparse
import os
import cv2
from transformers import BlipProcessor, BlipForConditionalGeneration
import json
import time
from local_config import device, voc_path, batch_size, limit_no_of_batches
from local_config import diffusion_timestamps as diffusion_timestamps_list
from local_config import run_type, prompts_dir, prompts_path, RunType
from open_vocabulary.get_model import get_diffusion_model

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

stop_flag_path = './stop.txt'

class LocalBlend:
    def __call__(self, x_t, attention_store):
        k = 1
        maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
        maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, MAX_NUM_WORDS) for item in maps]
        maps = torch.cat(maps, dim=1)
        maps = (maps * self.alpha_layers).sum(-1).mean(1)
        mask = nnf.max_pool2d(maps, (k * 2 + 1, k * 2 +1), (1, 1), padding=(k, k))
        mask = nnf.interpolate(mask, size=(x_t.shape[2:]))
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        mask = mask.gt(self.threshold)
        mask = (mask[:1] + mask[1:]).float()
        x_t = x_t[:1] + mask * (x_t - x_t[:1])
        return x_t
       
    def __init__(self, prompts: List[str], words: [List[List[str]]], threshold=.3):
        alpha_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1
        self.alpha_layers = alpha_layers.to(device)
        self.threshold = threshold


class AttentionControl(abc.ABC):
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0
    
    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

class EmptyControl(AttentionControl):
    
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        return attn
    
    
class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 64 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention


    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

        
class AttentionControlEdit(AttentionStore, abc.ABC):
    
    def step_callback(self, x_t):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)
        return x_t
        
    def replace_self_attention(self, attn_base, att_replace):
        if att_replace.shape[2] <= 16 ** 2:
            return attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
        else:
            return att_replace
    
    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError
    
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_repalce = attn[0], attn[1:]
            if is_cross:
                alpha_words = self.cross_replace_alpha[self.cur_step]
                attn_repalce_new = self.replace_cross_attention(attn_base, attn_repalce) * alpha_words + (1 - alpha_words) * attn_repalce
                attn[1:] = attn_repalce_new
            else:
                attn[1:] = self.replace_self_attention(attn_base, attn_repalce)
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn
    
    def __init__(self, prompts, num_steps: int,
                 cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                 self_replace_steps: Union[float, Tuple[float, float]],
                 local_blend: Optional[LocalBlend]):
        super(AttentionControlEdit, self).__init__()
        self.batch_size = len(prompts)
        self.cross_replace_alpha = ptp_utils.get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps, tokenizer).to(device)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.local_blend = local_blend

class AttentionReplace(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)
      
    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionReplace, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.mapper = seq_aligner.get_replacement_mapper(prompts, tokenizer).to(device)
        

class AttentionRefine(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionRefine, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.mapper, alphas = seq_aligner.get_refinement_mapper(prompts, tokenizer)
        self.mapper, alphas = self.mapper.to(device), alphas.to(device)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])


class AttentionReweight(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        if self.prev_controller is not None:
            attn_base = self.prev_controller.replace_cross_attention(attn_base, att_replace)
        attn_replace = attn_base[None, :, :, :] * self.equalizer[:, None, None, :]
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float, equalizer,
                local_blend: Optional[LocalBlend] = None, controller: Optional[AttentionControlEdit] = None):
        super(AttentionReweight, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.equalizer = equalizer.to(device)
        self.prev_controller = controller

class Mask:
    mask: torch.Tensor
    label: str

    def __init__(self, mask: torch.Tensor, label: str):
        self.mask = mask
        self.label = label

class ImageMasks:
    masks: List[Mask]
    image_sizes: Tuple[int, int]

    def __init__(self, image_sizes: Tuple[int, int]):
        self.masks = []
        self.image_sizes = image_sizes

    def add_mask(self, mask: Mask):
        self.masks.append(mask)


def get_equalizer(text: str, word_select: Union[int, Tuple[int, ...]], values: Union[List[float],
                  Tuple[float, ...]]):
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(len(values), 77)
    values = torch.tensor(values, dtype=torch.float32)
    for word in word_select:
        inds = ptp_utils.get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = values
    return equalizer
from PIL import Image

def aggregate_all_attention(prompts,attention_store: AttentionStore, from_where: List[str], is_cross: bool, select: int):
    attention_maps = attention_store.get_average_attention()
    att_8 = []
    att_16 = []
    att_32 = []
    att_64 = []
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if is_cross or item.shape[1] == 64**2:
                if item.shape[1] == 8*8:
                    cross_maps = item.reshape(len(prompts), -1, 8, 8, item.shape[-1])#[select]
                    att_8.append(cross_maps)
                if item.shape[1] == 16*16:
                    cross_maps = item.reshape(len(prompts), -1, 16, 16, item.shape[-1])#[select]
                    att_16.append(cross_maps)
                if item.shape[1] == 32*32:
                    cross_maps = item.reshape(len(prompts), -1, 32, 32, item.shape[-1])#[select]
                    att_32.append(cross_maps)
                if item.shape[1] == 64*64:
                    cross_maps = item.reshape(len(prompts), -1, 64, 64, item.shape[-1])#[select]
                    att_64.append(cross_maps)
    atts = []
    for att in [att_8,att_16,att_32,att_64]:
        if len(att) > 0:
            att = torch.cat(att, dim=1)
            att = att.sum(1) / att.shape[1]
            atts.append(att.cpu())
    return atts

def aggregate_attention(prompts,attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out).mean(0)
    return out.cpu()


# def show_cross_attention(prompts,attention_store: AttentionStore, res: int, from_where: List[str], select: int = 0): 
#     decoder = tokenizer.decode
#     attention_maps = aggregate_attention(prompts, attention_store, res, from_where, True, select)
#     images = []
#     texts = []
#     maps = []
#     tokens = tokenizer.encode(prompts[select])
#     map = []
#     for j in range(len(tokens)):
#         map.append(attention_maps[:,:,j])

#         texts.append(decoder(int(tokens[j])))
#     maps.append(torch.stack(map))
#     return images,texts,maps
#     # ptp_utils.view_images(np.stack(images, axis=0))
    

def show_self_attention_comp(prompts, attention_store: AttentionStore, res: int, from_where: List[str],
                        max_com=10, select: int = 0):
    attention_maps = aggregate_attention(prompts, attention_store, res, from_where, False, select).view(res ** 2, res ** 2)
    # u, s, vh = np.linalg.svd(attention_maps - np.mean(attention_maps, axis=1, keepdims=True))
    images = []
    # for i in range(max_com):
    #     image = vh[i].reshape(res, res)
    #     image = image - image.min()
    #     image = 255 * image / image.max()
    #     image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(np.uint8)
    #     image = Image.fromarray(image).resize((256, 256))
    #     image = np.array(image)
    #     images.append(image)
    return images, attention_maps
    # ptp_utils.view_images(np.concatenate(images, axis=1))

def run_and_display(ldm_stable, prompts, controller, latent=None, run_baseline=False, generator=None,t=100,noise_sample_num=2):
    if run_baseline:
        print("w.o. prompt-to-prompt")
        images, latent = run_and_display(ldm_stable, prompts, EmptyControl(), latent=latent, run_baseline=False, generator=generator)
        print("with prompt-to-prompt")
    images, x_t = ptp_utils.text2image_ldm_stable(ldm_stable, prompts, controller, latent=latent, num_inference_steps=t, guidance_scale=GUIDANCE_SCALE, generator=generator, low_resource=LOW_RESOURCE,noise_sample_num=noise_sample_num)
    # ptp_utils.view_images(images)
    # import ipdb;ipdb.set_trace()
    return images, x_t

def encode_imgs(imgs,vae):
    # imgs: [B, 3, H, W]
    imgs = 2 * imgs - 1
    posterior = vae.encode(imgs).latent_dist
    latents = posterior.sample() * 0.18215

    return latents

MY_TOKEN = None
LOW_RESOURCE = False 
NUM_DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77


def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=batch_size, type=int)

    parser.add_argument('--input-size', default=224, type=int, help='images input size')
    # Dataset parameters
    parser.add_argument('--data-path', default='', type=str, help='dataset path')
    parser.add_argument('--img-list', default='', type=str, help='image list path')
    parser.add_argument('--data-set', default='', type=str, help='dataset')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='mps',
                        help='device to use for training / testing')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    return parser

embs = [
        "plane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "buses",
        "car",
        "cat",
        "chair",
        "cow",
        "table",
        "dog",
        "horse",
        "motorbike",
        "people",
        "plant",
        "sheep",
        "sofa",
        "train",
        "monitor"
    ]

# srcfile 需要复制、移动的文件   
# dstpath 目的地址
 
import os
import shutil
from glob import glob
 
def mycopyfile(srcfile,dstpath):                       # 复制函数
    if not os.path.isfile(srcfile):
        print ("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(srcfile)             # 分离文件名和路径
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)                       # 创建路径
        shutil.copy(srcfile, f'{dstpath}/{fname}')          # 复制文件
        print ("copy %s -> %s"%(srcfile, f'{dstpath}/{fname}'))
                       
def read_json(json_path, choice):
    with open(json_path,"r") as tmp:
        data=json.load(tmp)
    re=data[choice[0]]
    if len(choice)==1:
        return re
    for key in re.keys():
        for i in range(1,len(choice)):
            re[key].extend(data[choice[i]][key])
        re[key]=list(set(re[key]))
    return re 

def same_seeds(seed):
    torch.manual_seed(seed)  # 固定随机种子（CPU）
    if torch.cuda.is_available():  # 固定随机种子（GPU)
        torch.cuda.manual_seed(seed)  # 为当前GPU设置
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置
    np.random.seed(seed)  # 保证后续使用random函数时，产生固定的随机数
    torch.backends.cudnn.benchmark = False  # GPU、网络结构固定，可设置为True
    torch.backends.cudnn.deterministic = True  # 固定网络结构

def compute_prompts(args, data_loader, batch_limiter):
    prompts_dict = {}
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    processor.tokenizer.padding_side = "left"
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)
    
    if not os.path.isdir(prompts_dir):
        os.mkdir(prompts_dir)

    for k,data in enumerate(data_loader):
        if k >= batch_limiter:
            break
        
        _, _, _, paths, _labels, _, image_ids, _ = data

        prompts = [f"a photograph of {current_label}" for current_label in _labels]
        raw_images = [Image.open(img_path).convert("RGB") for img_path in paths]

        inputs = processor(raw_images, prompts, return_tensors="pt", padding=True).to(device)
        out = model.generate(**inputs)
        out_prompts = processor.batch_decode(out, skip_special_tokens=True)

        prompts = [(prompt + "++" + out_prompt[len(prompt):]) for prompt, out_prompt in zip(prompts, out_prompts)]
        print(f"Prompt index: {k * args.batch_size}")

        for prompt, image_id, label in zip(prompts, image_ids, _labels):
            prompts_dict[f'{image_id}_{label}'] = prompt
    
    with open(prompts_path, "w") as f:
        f.write(json.dumps(prompts_dict, indent=4))

def load_prompts():
    if os.path.isfile(prompts_path):
        with open(prompts_path) as f:
            prompts_dict = json.load(f)
        return prompts_dict
    else:
        print(f"Did not find a prompts dict at {prompts_dict}")

def compute_masks(args, data_loader, batch_limiter):
    prompts_dict = load_prompts()
    ldm_stable, diffusion_timestamps = get_diffusion_model()

    ts_to_key_func = lambda x: '-'.join([str(entry) for entry in x])
    times_array: List[List[int]] = [[int(t * diffusion_timestamps) if (t * diffusion_timestamps)>1 else 1 for t in times_list] for times_list in diffusion_timestamps_list]
    print('Diffusion timestamps', diffusion_timestamps, times_array)
    
    output_path=f'./output/voc12_wo_sy_norm'
    img_output_path = f"{output_path}/images"
    # prediced image class label by clip and blip
    json_path='dataset/voc12/predict_syn_txt_0.8_img_0.97.json'
    # for paths in [img_output_path]:
    #     if not os.path.exists(paths):
    #         os.makedirs(paths)
    for current_time in times_array:
        current_time_path = os.path.join(img_output_path, ts_to_key_func(current_time))
        if not os.path.exists(current_time_path):
            os.makedirs(current_time_path)

    dtype = torch.float16
    vae = ldm_stable.vae.to(device)
    controller = AttentionStore()
    g_cpu = torch.Generator(4307)
    # TODO: best weights for LCM as well?
    weight = [0.3,0.5,0.1,0.1]
    
    print(len(data_loader))
    with torch.no_grad():
        imgs_with_pending_classes: Dict[str, Dict[str, ImageMasks]] = {}
        for _ts in times_array:
            imgs_with_pending_classes[ts_to_key_func(_ts)] = {}

        for k,data in enumerate(data_loader):
            if k >= batch_limiter:
                break
            
            rgb_512, _, _, paths, _labels, label_idxs, image_ids, raw_image_shapes = data
           
            rgb_512 = rgb_512 if dtype == torch.float32 else rgb_512.half()
            rgb_512 = rgb_512.to(device)
            input_latent = encode_imgs(rgb_512,vae).to(device)

            noise_sample_num = args.batch_size
            noise = torch.randn([args.batch_size,4,64,64]).to(device)
            noise = noise if dtype == torch.float32 else noise.half()

            prompts = [prompts_dict[f'{image_id}_{img_label}'] for image_id, img_label in zip(image_ids, _labels)]
            
            for current_timestamps in times_array:
                attn_maps = []
                for t in current_timestamps:
                    controller.reset()
                    latents_noisy = ldm_stable.scheduler.add_noise(input_latent, noise, torch.tensor([t] * args.batch_size, device=device))
                    latents_noisy = latents_noisy if dtype == torch.float32 else latents_noisy.half()
                    run_and_display(ldm_stable, prompts, controller, latents_noisy, run_baseline=False, generator=g_cpu, t=t, noise_sample_num=noise_sample_num)
                    
                    out_atts = []
                    cross_attention_maps = aggregate_all_attention(prompts,controller, ("up", "mid", "down"), True, 0)
                    self_attention_maps = aggregate_all_attention(prompts,controller, ("up", "mid", "down"), False, 0)

                    for idx,res in enumerate([8,16,32,64]):
                        if len(prompts[0].split(" "))>4 and prompts[0].split(" ")[4].endswith("ing"):
                            # average attn for words corresponding to tokens for word of interest and "ing" if it is the case
                            cross_att = cross_attention_maps[idx][:,:,:,[4,5]].mean(3).view(args.batch_size, res,res).float()
                        else:
                            cross_att = cross_attention_maps[idx][:,:,:,[4]].mean(3).view(args.batch_size, res,res).float()
                        if res != 64:
                            cross_att = F.interpolate(cross_att.unsqueeze(0), size=(64,64), mode='bilinear', align_corners=False).squeeze()
                            if len(cross_att.shape) < 3:
                                cross_att = cross_att.unsqueeze(0)
                        cross_att = (cross_att-cross_att.min())/(cross_att.max()-cross_att.min())
                        out_atts.append(cross_att * weight[idx])

                    cross_att_map = torch.stack(out_atts).sum(0).view(args.batch_size, 64*64, 1) # TODO: was 1 instead of bs, is it correct for bs != 1?
                    self_att = self_attention_maps[0].view(args.batch_size, 64*64, 64*64).float() # TODO: using only the 3rd one, use all or compute just 3rd
                    attn_maps.append(
                        torch.matmul(self_att,cross_att_map).view(args.batch_size, res, res)
                    )                
                average_attn_map = torch.mean(torch.stack(attn_maps, dim=0),dim=0).unsqueeze(1).detach().cpu().repeat(1,3,1,1)
                for i in range(len(paths)):
                    current_img_id = image_ids[i]
                    current_label = _labels[i]
                    current_label_idx = label_idxs[i]
                    current_mask = average_attn_map[i]
                    raw_image_size = raw_image_shapes[i] # rgb image height and width

                    current_timestamp_key = ts_to_key_func(current_timestamps)
                    pending_classes_for_current_ts: Dict[str, ImageMasks] = imgs_with_pending_classes[current_timestamp_key]
                    
                    if current_label_idx == 0:
                        # just processed the last class for current image
                        if current_img_id in pending_classes_for_current_ts:
                            # merge result with results on the other target classes, if image had more classes associated
                            image_masks = [mask.mask for mask in pending_classes_for_current_ts[current_img_id].masks] + [current_mask]
                            aggregated_masks = torch.stack(image_masks)
                            image_labels = [mask.label for mask in pending_classes_for_current_ts[current_img_id].masks] + [current_label]
                            original_image_sizes = pending_classes_for_current_ts[current_img_id].image_sizes
                        else:
                            aggregated_masks = current_mask.unsqueeze(0)
                            image_labels = [current_label]
                            original_image_sizes = raw_image_size

                        #aggregated_masks = aggregated_masks.to(device)
                        images = F.interpolate(aggregated_masks, size=(original_image_sizes[0].item(), original_image_sizes[1].item()), mode='bilinear', align_corners=False) # TODO: h,w for each img in batch??
                        for idx in range(images.shape[0]):                            
                            images[idx]=((images[idx]-images[idx].min())/(images[idx].max()-images[idx].min())) * 255
                        
                        cam_dict={}
                        for idx in range(images.shape[0]):
                            image_label = image_labels[idx]
                            cam_dict[str(embs.index(image_label))] = images[idx].permute(1,2,0).cpu().numpy()[:,:,0]
                        
                        save_path = os.path.join(img_output_path, current_timestamp_key, f'{current_img_id}.mat')
                        sio.savemat(save_path, cam_dict, do_compression=True)
                        if current_img_id in pending_classes_for_current_ts:
                            del pending_classes_for_current_ts[current_img_id]

                    else:
                        mask_object = Mask(current_mask, current_label)
                        if not current_img_id in pending_classes_for_current_ts:
                            pending_classes_for_current_ts[current_img_id] = ImageMasks(image_sizes=raw_image_size)
                        pending_classes_for_current_ts[current_img_id].add_mask(mask_object)
            
            if os.path.isfile(stop_flag_path):
                os.remove(stop_flag_path)
                break

            print(f"Image index: {k * args.batch_size}")

if __name__ == '__main__':
    same_seeds(4294)
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args(args=[])
    args.img_list='dataset/voc12'
    # if you want to run gt label, cls_labels_val.npy file has img class label
    args.label_file_path=None
    # your file path
    args.data_path= voc_path
    
    args.gen_attention_maps=False
    args.data_set='VOC12seg'
    dataset_train, args.nb_classes = build_dataset(is_train=False, args=args, is_big_data=True)
    sampler_train = torch.utils.data.SequentialSampler(dataset_train)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    cudnn.benchmark = True
    dtype = torch.float16
    _batch_limiter = limit_no_of_batches if limit_no_of_batches is not None else len(data_loader_train)
    g_cpu = torch.Generator(4307)

    start_time = time.time()
    with torch.no_grad():
        if run_type == RunType.PROMPTS or run_type == RunType.BOTH:
            compute_prompts(args, data_loader_train, _batch_limiter)
        if run_type == RunType.SEGMENTATIONS or run_type == RunType.BOTH:
            compute_masks(args, data_loader_train, _batch_limiter)

    print('Took me', int(time.time() - start_time))
